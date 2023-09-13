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

from pose_generator.pose_gen import generate_poses
from camera.basler_camera import BaslerCamera

CAMERA_CONNECTED = False

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
    if CAMERA_CONNECTED:
        cam = BaslerCamera(save_location="images")
        cam.connec
        cam.adjust_camera()

    

    poses = generate_poses(np.array([-200, -500, 0]), 500)
    success_count = 0
    for pose in poses:
        X, Y, Z, RA, RB, RC = pose
        handler.move_to_position_with_points(
            input, X=X, Y=Y, Z=Z, RA=90, RB=0, RC=90
        )
        operation = handler.check_point_fail_pass(input)
        print(operation)
        if operation:
            success_count += 1
            print(f"O:{operation}")
            if CAMERA_CONNECTED:
                cam.save_current_image()

        print("Pushing new pose")

    # transf, RX, RY, RZ, RA, RB, RC = handler.get_current_pos_base(input)
    handler.move_to_position_with_points(
        input, X=-200.0, Y=-500.0, Z=400.0, RA=90, RB=0, RC=90
    )
    print("Finished moving")
    print(f"Success count: {success_count}/{len(poses)}")
    client.disconnect()
