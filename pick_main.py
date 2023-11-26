# 1st party
import argparse
import os
import time
import json
from typing import Any, Tuple, Union
import copy
import requests

# 3rd party
from mlsocket import MLSocket
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Mine
from cluster.Client import get_megapose_estimation
from camera.basler_camera import BaslerCamera
from camera.Frame_processor import FrameProcessor
from KMR_IIWA.IIWA_robot import IIWA
from KMR_IIWA.IIWA_tools import IIWA_tools

LABELS = {
    1: "d01_controller",
    2: "d02_servo",
    3: "d03_main",
    4: "d04_motor",
    5: "d05_axle_front",
    6: "d06_battery",
    7: "d07_axle_rear",
    8: "d08_chassis",
}
LABELS_NUMS_KEY = [x + 48 for x in LABELS.keys()]  # ASCII code for numbers from 1 to 8

np.set_printoptions(suppress=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Pick and place")
    parser.add_argument("-h", "--host", type=str, default="10.35.129.250")
    parser.add_argument("-p", "--port", type=int, default=65432)

    return parser.parse_args()


def convert2TransfMatrix(TX, TY, TZ, A, B, C):
    RotationMatrix = R.from_euler("zyx", [A, B, C], degrees=True)
    TranslationMatrix = np.array([TX, TY, TZ]).reshape(
        3,
    )
    TransformationMatrix = np.eye(4)

    TransformationMatrix[:3, :3] = RotationMatrix.as_matrix()
    TransformationMatrix[:3, 3] = TranslationMatrix
    return TransformationMatrix


def translquat2transf(translation, quaternion, scale=1):
    RotationMatrix = R.from_quat(quaternion)

    # varun edit
    # rot_angle = R.from_matrix(RotationMatrix.as_matrix()).as_euler("xyz", degrees=True)
    # print(rot_angle)
    # RotationMatrix = R.from_euler("zyx", rot_angle, degrees=True)
    # TranslationMatrix = (
    # np.array(translation).reshape(
    # 3,
    # )
    # * scale
    # )
    translation_vector = np.array(translation).reshape(3) * scale

    TransformationMatrix = np.eye(4)

    TransformationMatrix[:3, :3] = RotationMatrix.as_matrix()
    TransformationMatrix[:3, 3] = translation_vector
    return TransformationMatrix


def translaxisangle2transf(translation, axis, angle):
    RotationMatrix = R.from_rotvec(axis * angle)
    TranslationMatrix = np.array(translation).reshape(
        3,
    )
    TransformationMatrix = np.eye(4)

    TransformationMatrix[:3, :3] = RotationMatrix.as_matrix()
    TransformationMatrix[:3, 3] = TranslationMatrix
    return TransformationMatrix


def transf2TxTyTzABC(transf):
    Rot = transf[:3, :3]
    Tra = transf[:3, 3]

    angles = R.from_matrix(Rot).as_euler("ZYX", degrees=True)
    A = angles[0]
    B = angles[1]
    C = angles[2]
    X = Tra[0]
    Y = Tra[1]
    Z = Tra[2]

    return X, Y, Z, A, B, C


def frame_processing_test():
    """Test function for FrameProcessor class"""
    detector = FrameProcessor()
    detector.connect()
    detector.adjust_camera()

    while True:
        should_quit, frame, bbox, idx = detector.proccess_frame()
        if should_quit:
            break
        if frame is not None or bbox is not None or idx is not None:
            time.sleep(5)

    cv2.destroyAllWindows()
    detector.disconnect()


def movement_test():
    """Test function for movement of the robot"""
    # recorded_path = os.path.join("pose_data", "d03_main.json")
    # with open(recorded_path, "r") as f:
    #     data = json.load(f)

    # print(data)

    # pose = np.array(data["pose"])
    # print(pose)

    robot1_ip = "172.31.1.10"
    robot1 = IIWA(robot1_ip)

    # create the tools
    camera = IIWA_tools(TX=50, TY=0, TZ=0, A=0, B=0, C=0, name="camera")
    gripper = IIWA_tools(TX=0, TY=0, TZ=200, A=0, B=0, C=0, name="gripper")

    iiwa_camera = IIWA_tools(TX=3, TY=-90, TZ=-15, A=0, B=0, C=0, name="camera")
    iiwa_gripper = IIWA_tools(
        TX=0, TY=0, TZ=230, A=0, B=0, C=0, name="gripper"
    )  # Tz = 230 for now 226 for touching the table

    # attach tool to the flange of the robot
    robot1.addTool(iiwa_camera)
    robot1.addTool(iiwa_gripper)

    robot1.closeGripper(position=0)
    robot1.sendCartisianPosition(
        X=-200, Y=-500, Z=500, A=90, B=0, C=180, motion="ptp", tool=None
    )
    # robot1.sendCartisianPosition(
    #     X=-100, Y=-550, Z=100, A=90, B=0, C=180, motion="ptp", tool=iiwa_gripper
    # )

    robot1.sendCartisianPosition(
        X=-200, Y=-500, Z=500, A=90, B=0, C=180, motion="ptp", tool=None
    )

    robot1.sendCartisianPosition(
        X=-200, Y=-500, Z=500, A=90, B=0, C=180, motion="ptp", tool=None
    )

    robot1.wait2ready()

    # robot1.sendCartisianPosition(X=-200, Y=-500, Z=300, A=90, B=0, C=180, motion='ptp', tool=gripper)
    # robot1.sendCartisianPosition(X=-200, Y=-500, Z=300, A=180, B=0, C=180, motion='ptp', tool=gripper)
    # robot1.openGripper()
    # robot1.sendCartisianPosition(X=-200, Y=-500, Z=50, A=180, B=0, C=180, motion='ptp', tool=gripper)
    # robot1.closeGripper(position=23000)
    # robot1.sendCartisianPosition(X=-200, Y=-500, Z=300, A=90, B=0, C=180, motion='ptp', tool=gripper)
    # robot1.wait2ready() # THis will ensure that the robot is ready to go to the next position


def print_transf(T):
    t = T[:3, 3]
    A, B, C = R.from_matrix(T[:3, :3]).as_euler("zyx", degrees=True)

    print(f"Transformation in x y z [mm] and rotation in ZYX euler:")
    print(
        f"\tTx: {t[0]:.2f}, Ty: {t[1]:.2f}, Tz: {t[2]:.2f}, A: {A:.2f}, B: {B:.2f}, C: {C:.2f}"
    )


def debug_transformations():
    # Flange
    f_Tx, f_Ty, f_Tz = [-200, -500, 500]
    f_A, f_B, f_C = [90, 0, 180]
    T_W2F = convert2TransfMatrix(f_Tx, f_Ty, f_Tz, f_A, f_B, f_C)
    print_transf(T_W2F)

    # Camera from Flange
    c_A, c_B, c_C = [0.45479412757838633, -0.8090880725702911, -0.5027631332617288]
    c_TX, c_TY, C_TZ = [6.570145950404226, -91.7642682003743, -13.340081274755768]
    T_C2F = convert2TransfMatrix(c_TX, c_TY, C_TZ, c_A, c_B, c_C)
    T_F2C = np.linalg.inv(T_C2F)
    print_transf(T_F2C)

    T_W2C = T_W2F @ T_F2C
    print("T_W2C")
    print_transf(T_W2C)  # THIS IS PROBABLY OK
    print("-")
    # Camera from Ob
    megapose_dict = json.load(open("pose_data/d03_main.json", "r"))
    megapose_pose = megapose_dict["pose"]
    q_C2Ob = megapose_pose[:4]
    t_C2Ob = megapose_pose[4:]
    T_C2Ob = translquat2transf(translation=t_C2Ob, quaternion=q_C2Ob, scale=1000)

    # T_C2Ob = np.linalg.inv(T_C2Ob)

    # T_W2Ob = T_W2F @ T_F2C @ T_C2Ob
    # print_transf(T_W2Ob)
    T_W2Ob = T_W2C @ T_C2Ob
    print_transf(T_W2Ob)

    # #Ob from Og
    main_trl = np.array([-2.9, 18.7, 14.27])
    main_axis = np.array([0.59, 0.58, -0.56])
    main_angle = 121.76
    T_Ob2Og = translaxisangle2transf(main_trl, main_axis, main_angle)
    # T_Ob2Og = np.linalg.inv(T_Ob2Og)

    T_x180 = np.eye(4)
    T_x180[:3, :3] = R.from_euler("zyx", [0, 0, 180], degrees=True).as_matrix()

    T_W2Og = T_W2F @ T_F2C @ T_C2Ob @ T_Ob2Og @ T_x180
    # print_transf(T_W2Og)
    print("fin")
    print_transf(T_W2Og)


def server_test():
    ml_socket = MLSocket()
    host = "10.35.129.250"
    port = 65432
    ml_socket.connect((host, port))
    print(f"Connection to {host}:{port} established")
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    frame = cv2.imread("server_test/Cybertech.png")
    print(frame.shape)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)

    K_dict = json.load(open("server_test/camera_Cybertech.json", "r"))

    K = np.array(K_dict["K"])
    K = K[:3, :3]
    dist_coef = np.array(K_dict["dist_coef"])
    h, w = frame.shape[:2]
    print(h, w)
    print(K_dict["resolution_distorted"])

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coef, (w, h), 1, (w, h))
    x, y, w, h = roi
    print(new_K)

    frame = cv2.undistort(frame, K, dist_coef, None, new_K)
    frame = frame[y : y + h, x : x + w]
    print(frame.shape)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)

    # new_K[0,0] /= 2
    # new_K[1,1] /= 2

    # new_K[0,2] /= 2
    # new_K[1,2] /= 2

    # frame = cv2.resize(frame, (int(w/2), int(h/2)))

    bbox = np.array([891, 685, 1095, 852])
    idx = np.array([3])

    cv2.imshow("frame", frame)
    cv2.waitKey(0)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # NOTE Megapose expects RGB
    # frame_rgb = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    pose = get_megapose_estimation(ml_socket, frame_rgb, bbox, idx, new_K)

    print(pose)

    with open("server_test/pose.json", "w") as f:
        json.dump(pose.tolist(), f, indent=2)

    _ = get_megapose_estimation(ml_socket, frame_rgb, bbox, np.array([-1]), new_K)

    ml_socket.close()


# THIS SHOULD BE THE MAIN FUNCTION
def main(robot_on: bool = True, server_on: bool = True):
    # Camera init
    print("Initializing camera")
    detector = FrameProcessor(camera_parametes="camera/camera_params.json")
    detector.connect()
    detector.adjust_camera()
    # detector.camera
    print("Camera initialized")
    K = detector.camera_ideal_params["K"]

    with open(os.path.join("camera", "extrinsic_calibration.json"), "r") as f:
        extrinsic_calibration = json.load(f)

    t_f2c = np.array(extrinsic_calibration["t_C2F"])
    c_angles = np.array(extrinsic_calibration["angles"])

    # Camera Extrinsic parameters # TODO: REDO THE INTRINSIC AND EXRTINSIC PARAMETERS
    c_A, c_B, c_C = t_f2c.flatten().tolist()
    c_TX, c_TY, c_TZ = c_angles.flatten().tolist()
    iiwa_camera = IIWA_tools(
        TX=c_TX, TY=c_TY, TZ=c_TZ, A=c_A, B=c_B, C=c_C, name="camera"
    )
    # Maybe add loading from file
    g_TX, g_TY, g_TZ = [5, -15, 230]
    g_A, g_B, g_C = [0, 0, 0]
    # iiwa_gripper = IIWA_tools(TX=5, TY=-15, TZ=230, A=0, B=0, C=0, name="gripper")
    iiwa_gripper = IIWA_tools(
        TX=g_TX, TY=g_TY, TZ=g_TZ, A=g_A, B=g_B, C=g_C, name="gripper"
    )
    prepick_z_offset = 100
    iiwa_prepick = IIWA_tools(
        TX=g_TX,
        TY=g_TY,
        TZ=g_TZ + prepick_z_offset,
        A=g_A,
        B=g_B,
        C=g_C,
        name="prepick",
    )

    # Viewing position
    v_Tx, v_Ty, v_Tz = [-200, -500, 500]
    v_A, v_B, v_C = [90, 0, 180]

    print("Initializing robot")
    if robot_on:
        robot1_ip = "172.31.1.10"
        iiwa = IIWA(robot1_ip)
        iiwa.addTool(iiwa_camera)
        iiwa.addTool(iiwa_gripper)
        iiwa.addTool(iiwa_prepick)

        pos = iiwa.getCartisianPosition(tool=None)
        iiwa.openGripper()

        iiwa.sendCartisianPosition(
            X=v_Tx, Y=v_Ty, Z=v_Tz, A=v_A, B=v_B, C=v_C, motion="ptp", tool=None
        )

    # TODO: Also intrinsics
    # camera_translation = np.array([1, -88.5, 55])
    # camera_rotation = np.array([0, 0, 0])
    camera_translation = t_f2c
    camera_rotation = c_angles

    # Flange to camera transformation is static basically
    T_F2C = np.eye(4)
    T_F2C[:3, :3] = R.from_euler("zyx", camera_rotation, degrees=True).as_matrix()
    T_F2C[:3, 3] = camera_translation.flatten()

    # print("world postion in camera frame")
    # print(camera_position)

    # TODO: Add parser for the host and port
    # Server comunication init
    if server_on:
        ml_socket = MLSocket()
        host = "10.35.129.250"
        port = 65432
        ml_socket.connect((host, port))
        print(f"Connection to {host}:{port} established")

    # Main loop
    while True:
        should_quit, frame, bbox, idx = detector.proccess_frame()
        if should_quit:  # Should quit after q is pressed in the window
            # This will sent data to server to stop the server and close the connection (idx = -1)
            if server_on:  # Turns off the server
                _ = get_megapose_estimation(
                    ml_socket, np.zeros((3, 3, 3)), np.zeros((4)), np.array([-1]), K
                )
            break

        # While the user do not provide all the data the loop will continue
        if frame is None or bbox is None or idx is None:
            continue

        # Sanitazing the data for the communication with the server
        if (
            type(frame) != np.ndarray
            or type(bbox) != np.ndarray
            or type(idx) != np.ndarray
        ):
            print("Wrong type try again")
            detector.reset()
            continue

        # Megapose Cluster estimation
        if server_on:
            frame_rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )  # NOTE Megapose expects RGB
            pose = get_megapose_estimation(ml_socket, frame_rgb, bbox, idx, K)

            #
            # Saving the data
            print(
                f"Data received from megapose cluster:\n\tquat:{pose[:4]}\n\ttrnl:{pose[4:]}"
            )
            recorded_data = {
                "label": LABELS[idx[0]],
                "bbox": bbox.tolist(),
                "idx": idx.tolist(),
                "pose": pose.tolist(),
            }
            cv2.imwrite(os.path.join("pose_data", f"{LABELS[idx[0]]}.png"), frame)
            with open(os.path.join("pose_data", f"{LABELS[idx[0]]}.json"), "w") as f:
                json.dump(recorded_data, f, indent=2)
        else:
            pose = np.array(
                [
                    0.666650741493408,
                    -0.01583200306880535,
                    -0.000888065233182203,
                    0.7452015372812825,
                    -0.0036940101999789476,
                    -0.014647329226136208,
                    0.35705795884132385,
                ]
            )  # This is hardcoded for now TODO: remove this or replace with the last recorded data

        if robot_on:
            robot_flange_loc_dict = iiwa.getCartisianPosition(tool=None)
            v_Tx_robot = robot_flange_loc_dict["x"]
            v_Ty_robot = robot_flange_loc_dict["y"]
            v_Tz_robot = robot_flange_loc_dict["z"]

            v_A_robot = robot_flange_loc_dict["A"]
            v_B_robot = robot_flange_loc_dict["B"]
            v_C_robot = robot_flange_loc_dict["C"]
        else:
            v_Tx_robot, v_Ty_robot, v_Tz_robot = v_Tx, v_Ty, v_Tz
            v_A_robot, v_B_robot, v_C_robot = v_A, v_B, v_C

        # Cureent location of the robot flange
        T_W2F = np.eye(4)
        T_W2F[:3, 3] = np.array([v_Tx_robot, v_Ty_robot, v_Tz_robot])
        T_W2F[:3, :3] = R.from_euler(
            "ZYX", [v_A_robot, v_B_robot, v_C_robot], degrees=True
        ).as_matrix()

        T_C2Ob = np.eye(4)
        T_C2Ob[:3, :3] = R.from_quat(
            np.array([pose[0], pose[1], pose[2], pose[3]])
        ).as_matrix()
        T_C2Ob[:3, 3] = pose[4:] * 1000  # m2mm

        # T_W2Ob = T_W2F @ T_F2C @ T_C2Ob

        # This could be also loaded from some file
        T_F2G = np.eye(4)
        T_F2G[:3, 3] = np.array([5, -15, 230])

        T_F2P = np.eye(4)
        T_F2P[:3, 3] = np.array([5, -15, 330])

        T_Ob2Og = np.eye(4)
        # TODO: Add the base to grip transformation here for all the objects
        # TODO: Update the IIWA tools to the np.linag.inv(T_F2G) inside the class

        T_W2Og = T_W2F @ T_F2C @ T_C2Ob @ T_Ob2Og
        grip_rotation = R.from_matrix(T_W2Og[:3, :3]).as_euler("ZYX", degrees=True)
        grip_translation = T_W2Og[:3, 3]
        Og_TX = grip_translation[0]
        Og_TY = grip_translation[1]
        Og_TZ = grip_translation[2]
        Og_A = grip_rotation[0]
        Og_B = grip_rotation[1]
        Og_C = grip_rotation[2]


        # T_W2Og = T_W2F @ T_F2C @ T_C2Ob @ T_Ob2Og @ np.linalg.inv(T_F2G)
        # T_W2Op = T_W2F @ T_F2C @ T_C2Ob @ T_Ob2Og @ np.linalg.inv(T_F2P)

        # prepick_rotation = R.from_matrix(T_W2Op[:3, :3]).as_euler("ZYX", degrees=True)
        # prepick_translation = T_W2Op[:3, 3]
        # Op_TX = prepick_translation[0]
        # Op_TY = prepick_translation[1]
        # Op_TZ = prepick_translation[2]
        # Op_A = prepick_rotation[0]
        # Op_B = prepick_rotation[1]
        # Op_C = prepick_rotation[2]

        # pic_rotation = R.from_matrix(T_W2Og[:3, :3]).as_euler("ZYX", degrees=True)
        # pic_translation = T_W2Og[:3, 3]
        # Og_TX = pic_translation[0]
        # Og_TY = pic_translation[1]
        # Og_TZ = pic_translation[2]
        # Og_A = pic_rotation[0]
        # Og_B = pic_rotation[1]
        # Og_C = pic_rotation[2]

        if robot_on:
            iiwa.openGripper()
            # print("Sending to prepick position")
            # succes_report = iiwa.sendCartisianPosition(
            #     X=Op_TX,
            #     Y=Op_TY,
            #     Z=Op_TZ,
            #     A=Op_A,
            #     B=Op_B,
            #     C=Op_C,
            #     motion="ptp",
            #     tool=None,
            # )
            # print(succes_report)

            # print("Sending to pick position")
            # succes_report = iiwa.sendCartisianPosition(
            #     X=Og_TX,
            #     Y=Og_TY,
            #     Z=Og_TZ,
            #     A=Og_A,
            #     B=Og_B,
            #     C=Og_C,
            #     motion="ptp",
            #     tool=None,
            # )

            # print(succes_report)
            print("Sending to prepick position")
            succes_report = iiwa.sendCartisianPosition(
                X=Og_TX,
                Y=Og_TY,
                Z=Og_TZ,
                A=Og_A,
                B=Og_B,
                C=Og_C,
                motion="ptp",
                tool=iiwa_prepick,
            )
            print(succes_report)
            print("Sending to pick position")
            succes_report = iiwa.sendCartisianPosition(
                X=Og_TX,
                Y=Og_TY,
                Z=Og_TZ,
                A=Og_A,
                B=Og_B,
                C=Og_C,
                motion="ptp",
                tool=iiwa_gripper,
            )

            print(iiwa.getCartisianPosition(tool=None))

            # NOTE: For now
            # d08_chassis == 27000
            # d03_main == 50000
            iiwa.closeGripper(
                position=50000
            )  # TODO: ADD the gripper position based on the object in the future based on the gripping position

            print("Sending to home (viewing) position")
            succes_report = iiwa.sendCartisianPosition(
                X=v_Tx, Y=v_Ty, Z=v_Tz, A=v_A, B=v_B, C=v_C, motion="ptp", tool=None
            )
            print(succes_report)

        detector.reset()
        print("Finished movement \nReady for next object")

    # Deactivate everything else
    print("Quitting the program")

    detector.disconnect()
    cv2.destroyAllWindows()

    if server_on:
        ml_socket.close()


if __name__ == "__main__":
    main(robot_on=True, server_on=True)
    # frame_processing_test()
    # movement_test()
    # debug_transformations()
    # server_test()
