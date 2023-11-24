import os
import json

import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":
    path = os.path.join("camera", "new_capture_ext")
    files = sorted(os.listdir(path))
    print(files)

    with open(os.path.join("camera", "camera_params.json")) as f:
        camera_parameters = json.load(f)

    print(camera_parameters)

    K = np.array(camera_parameters["camera_matrix"])
    dist_coeffs = np.array(camera_parameters["distortion_coefficients"])

    marker_length = 174.75  # mm

    R_gripper2base = []
    t_gripper2base = []

    R_target2cam = []
    t_target2cam = []
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    for json_file, image_file in zip(files[0::2], files[1::2]):
        with open(os.path.join(path, json_file), "r") as f:
            pos = json.load(f)

        A, B, C = pos["A"], pos["B"], pos["C"]
        x, y, z = pos["x"], pos["y"], pos["z"]

        # Base to flange/gripper
        T_B2F = np.eye(4)
        T_B2F[:3, 3] = np.array([x, y, z])
        T_B2F[:3, :3] = R.from_euler("ZYX", [A, B, C], degrees=True).as_matrix()
        print(R.from_matrix(T_B2F[:3, :3]).as_euler("ZYX", degrees=True))


        # Flange/gripper to base
        T_F2B = np.linalg.inv(T_B2F)
        R_F2B = T_F2B[:3, :3]
        t_F2B = T_F2B[:3, 3]


        R_gripper2base.append(T_B2F[:3, :3])
        t_gripper2base.append(T_B2F[:3, 3])


        # Now for the camera
        image = cv2.imread(os.path.join(path, image_file))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params
        )

        if len(corners) > 0:
            for i in range(len(ids)):
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], marker_length, K, dist_coeffs
                )
                cv2.drawFrameAxes(image, K, dist_coeffs, rvec, tvec, marker_length/2)


                print(tvec)


                R_C2T, _ = cv2.Rodrigues(rvec)
                t_C2T = tvec
                T_C2T = np.eye(4)

                # T_C2T[:3, :3] = R_C2T
                # T_C2T[:3, 3] = t_C2T.flatten()

                # T_T2C = np.linalg.inv(T_C2T)
                # R_T2C = T_T2C[:3, :3]
                # t_T2C = T_T2C[:3, 3]

                R_T2C = R_C2T
                t_T2C = t_C2T.flatten()

                print(R_T2C)
                print(t_T2C)

                R_target2cam.append(R_T2C)
                t_target2cam.append(t_T2C)

        cv2.circle(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)), 5, (255, 0, 255), -1)
        cv2.imshow("frame", image)

        key = cv2.waitKey(0)
        if key == ord("q"):
            break


    print("Calibrating...")
    R_gripper2base = np.array(R_gripper2base)
    t_gripper2base = np.array(t_gripper2base)
    R_target2cam = np.array(R_target2cam)
    t_target2cam = np.array(t_target2cam)

    print(R_gripper2base.shape)
    print(R_target2cam.shape)

    print(t_gripper2base.shape)
    print(t_target2cam.shape)

    R_C2F, t_C2F = cv2.calibrateHandEye(
        R_gripper2base=np.array(R_gripper2base),
        t_gripper2base=np.array(t_gripper2base),
        R_target2cam=np.array(R_target2cam),
        t_target2cam=np.array(t_target2cam),
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    print("Final results")
    print(R.from_matrix(R_C2F).as_euler("ZYX", degrees=True))
    # print(R_C2F)
    print(t_C2F)

    angles = R.from_matrix(R_C2F).as_euler("ZYX", degrees=True)
    print(angles)
    dict = {
        "R_C2F": R_C2F.tolist(),
        "t_C2F": t_C2F.tolist(),
        "angles": angles.tolist(),
    }
    with open(os.path.join("camera", "extrinsic_calibration.json"), "w") as f:
        json.dump(dict, f, indent=2)
