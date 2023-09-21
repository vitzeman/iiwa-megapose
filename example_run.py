import logging
import time
from opcua.common.node import Node
from opcua import Client
from pypylon import pylon
import cv2
import os
import argparse
import numpy as np
import json
from utils.robot_motion import path, SubHandler
from matplotlib import pyplot as plt

# get instance of the pylon TransportLayerFactory
tlf = pylon.TlFactory.GetInstance()
devices = tlf.EnumerateDevices()
bop_cam = None

serial_number = "24380112"

for d in devices:
    if d.GetSerialNumber() == serial_number:
        bop_cam = d

# camera = pylon.InstantCamera(tlf.CreateDevice(bop_cam))
# camera.Open()

# ## Set things to auto for best image possible
# camera.GainAuto.SetValue("Once")
# camera.ExposureAuto.SetValue("Once")
# camera.BalanceWhiteAuto.SetValue("Once")

# camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
# converter = pylon.ImageFormatConverter()
# converter.OutputPixelFormat = pylon.PixelType_BGR8packed

from pose_generator.pose_gen import generate_poses, nice_pose_print, visualize_pose
from camera.basler_camera import BaslerCamera

CAMERA_CONNECTED = True

if __name__ == "__main__":
    client = Client("opc.tcp://localhost:5000/")
    client.connect()
    client.load_type_definitions()
    root = client.get_root_node()
    objects = client.get_objects_node()
    uri = "http://iiwa-control.ciirc.cz"
    idx = client.get_namespace_index(uri)
    rootObj = objects.get_child(["IIWA"])
    input = rootObj.get_child(["RobotGenericInput"])
    output = rootObj.get_child(["RobotGenericOutput"])
    handler = SubHandler()
    subscription = client.create_subscription(100, handler)
    subscription.subscribe_data_change(
        [
            input.get_child(["ActualX"]),
            input.get_child(["ActualY"]),
            input.get_child(["ActualZ"]),
            input.get_child(["ActualRA"]),
            input.get_child(["ActualRB"]),
            input.get_child(["ActualRC"]),
            input.get_child(["OperationStarted"]),
            input.get_child(["OperationFinished"]),
            input.get_child("Failed"),
        ]
    )

    subscription.subscribe_events(rootObj.get_child(["InMotionEvent"]))

    subscription.subscribe_events(rootObj.get_child(["FailEvent"]))

    # Home position from myin.py
    # handler.move_to_position_with_points(input, X=550, Y=0, Z=400, RA=360, RB=0, RC=270)

    # handler.move_to_position_with_points(input, X=30, Y=0, Z=-400, RA=180, RB=0, RC=90)

    # This works for some reason
    # handler.move_to_position_with_points(
    #     input, X=-200.0, Y=-500.0, Z=500.0, RA=90, RB=0, RC=90
    # )
    calibration_num = "calibration-180"
    if CAMERA_CONNECTED:
        cam = BaslerCamera(
            save_location=os.path.join("camera", calibration_num, "images")
        )
        cam.connect()
        cam.adjust_camera()

    center = [0, -550, 40]
    poses = generate_poses(
        np.array(center),
        500,
        Rz=-90,
        phi_gen=range(0, 360, 15),
        y_limits=(-1000, -200),
        z_limits=(100, 600),
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot([center[0]], [center[1]], [center[2]], marker="x", color="r")
    ax.quiver(0, 0, 0, 1, 0, 0, length=100, color="r", linewidth=5)
    ax.quiver(0, 0, 0, 0, 1, 0, length=100, color="g", linewidth=5)
    ax.quiver(0, 0, 0, 0, 0, 1, length=100, color="b", linewidth=5)

    for pose in poses:
        ax.scatter(pose[0], pose[1], pose[2], marker="o")
        vector = center - np.array([pose[0], pose[1], pose[2]])
        vector = vector / np.linalg.norm(vector)
        ax.quiver(
            pose[0],
            pose[1],
            pose[2],
            vector[0],
            vector[1],
            vector[2],
            length=100,
            color="k",
        )
        ax = visualize_pose(pose, ax)
    # plt.show()
    plt.savefig(os.path.join("camera", calibration_num, "poses.png"))

    poses_dict = {"generated_poses": poses}
    reached_poses = []

    success_count = 0
    for e, pose in enumerate(poses):
        print("Sending to new pose:", end="\t")
        nice_pose_print(pose)
        X, Y, Z, RZ, RY, RX = pose
        handler.move_to_position_with_points(input, X=X, Y=Y, Z=Z, RA=RZ, RB=RY, RC=RX)
        operation = handler.check_point_fail_pass(input)
        print(operation)
        if operation:
            success_count += 1
            print(f"O:{operation}")
            transf_mtx, RX, RY, RZ, RA, RB, RC = handler.get_current_pos_base(input)
            pose_name = str(e).zfill(3)
            if CAMERA_CONNECTED:
                print("saving image")
                time.sleep(0.2)
                cam.save_current_image(name=pose_name + ".png")

            pose_data = {
                "W2C": transf_mtx.tolist(),
                "RX": RX,
                "RY": RY,
                "RZ": RZ,
                "RA": RA,
                "RB": RB,
                "RC": RC,
            }

            poses_dict[pose_name] = pose_data
            reached_poses.append(e)

        poses_dict["reached_poses"] = reached_poses

        print("Pushing new pose")

    with open(os.path.join("camera", calibration_num, "recorded_poses.json"), "w") as f:
        json.dump(poses_dict, f, indent=2)

    print("Finished moving - sending to prep position")
    transf, RX, RY, RZ, RA, RB, RC = handler.get_current_pos_base(input)
    handler.move_to_position_with_points(
        input, X=-40.0, Y=-750.0, Z=400.0, RA=-90, RB=0, RC=180
    )
    operational = handler.check_point_fail_pass(input)
    # handler.move_to_position_with_points(
    #     input, X=-200.0, Y=-500.0, Z=300.0, RA=-90, RB=0, RC=180
    # )
    print("Finished moving")
    # print(f"Success count: {success_count}/{len(poses)}")
    client.disconnect()

    if CAMERA_CONNECTED:
        cam.disconnect()
