import json
import os
from typing import Tuple, Union
import copy

import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation


def draw_base(img, corners, image_points):
    corner = tuple(corners[0].ravel().astype(int))

    image = copy.deepcopy(img)

    image = cv2.line(
        image, corner, tuple(image_points[0].ravel().astype(int)), (255, 0, 0), 5
    )
    image = cv2.line(
        image, corner, tuple(image_points[1].ravel().astype(int)), (0, 255, 0), 5
    )
    image = cv2.line(
        image, corner, tuple(image_points[2].ravel().astype(int)), (0, 0, 255), 5
    )
    return image


def get_image_pose(
    folder: str,
    name: str,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    num_verticies: Tuple[int, int] = (5, 8),
    square_size: float = 30,
    visualize: bool = False,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Get target to camera pose

    Args:
        folder (str): Folder with folder with images and pose.json
        name (str): Name of the image
        K (np.ndarray): _description_
        dist_coeffs (np.ndarray): Distortion coefficients
        num_verticies (Tuple[int, int], optional): Number of verticies in the chessboard. Defaults to (6, 8).
        square_size (float, optional): Size of the square in mm. Defaults to 30 mms.
        visualize (bool, optional): Visualize the pose. Defaults to False.

    Returns:
        Tuple[bool, np.ndarray, np.ndarray]: Success of detection, rotation and translation
    """

    path2img = os.path.join(folder, "images", name + ".png")
    img = cv2.imread(path2img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if visualize:
        img = cv2.putText(
            img, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    # cv2.imshow("img", img)

    ret, corners = cv2.findChessboardCorners(gray, num_verticies, None)
    if not ret:
        return False, None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    object_points = np.zeros((num_verticies[0] * num_verticies[1], 3), np.float32)
    # To get the units in mm multiply the distance by 30 plus the third coordinte is 0

    object_points[:, :2] = (
        np.mgrid[0 : num_verticies[0], 0 : num_verticies[1]].T.reshape(-1, 2)
        * square_size
    )

    ret, rvec, tvec = cv2.solvePnP(object_points, corners_subpix, K, dist_coeffs)
    R, _ = cv2.Rodrigues(rvec)

    if visualize:
        axis = (
            np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3) * square_size
        )
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, K, dist_coeffs)

        image = draw_base(img, corners_subpix, imgpts)
        cv2.imshow("img", image)
        cv2.waitKey(0)
    return True, R, tvec


def extrensic_calibration(folder) -> np.ndarray:
    """Computes the extrensic calibration of the camera.

    Args:
        folder (str): Path to folder with the image folder and pose.json.

    Returns:
        np.ndarray: _description_
    """
    with open(os.path.join(folder, "recorded_poses.json")) as f:
        poses = json.load(f)

    with open("camera_parameters.json") as f:
        camera_parameters = json.load(f)

    K = np.array(camera_parameters["camera_matrix"])
    dist_coeffs = np.array(camera_parameters["distortion_coefficients"])

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    reached = poses["reached_poses"]
    reached_names = [str(i).zfill(3) for i in reached]
    print(reached_names)
    for name in tqdm(reached_names, desc="Processing images", unit="image"):
        T_g2b = np.array(poses[name]["W2C"])
        T_g2b = np.linalg.inv(T_g2b)

        R_g2b = T_g2b[:3, :3]
        t_g2b = T_g2b[:3, 3]

        success, R_t2c, t_t2c = get_image_pose(folder, name, K, dist_coeffs)

        # Skips if the pose was not detected in the image(not fully in the image)
        if not success:
            print(f"Pose {name} was not detected")
            continue

        R_target2cam.append(R_t2c)
        t_target2cam.append(t_t2c)

        R_gripper2base.append(R_g2b)
        t_gripper2base.append(t_g2b)

    R_gripper2base, t_gripper2base = np.array(R_gripper2base), np.array(t_gripper2base)
    R_target2cam, t_target2cam = np.array(R_target2cam), np.array(t_target2cam)

    if len(R_gripper2base) <= 3:
        return None
    R_cam2flange, t_cam2flange = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    # convert to degrees
    angles = Rotation.from_matrix(R_cam2flange).as_euler("zyx", degrees=True)
    print(angles)
    print(t_cam2flange.flatten())

    transformation_dict = {
        "zyx_angles": angles.tolist(),
        "translation": t_cam2flange.flatten().tolist(),
        "Rotation_matrix": R_cam2flange.tolist(),
    }

    # Compute reprojection error
    mean = 0
    for i in range(len(R_gripper2base)):
        imgpoints2, _ = cv2.projectPoints(
            R_target2cam[i], R_gripper2base[i], t_gripper2base[i], K, dist_coeffs
        )
        print(imgpoints2.shape)

        error = cv2.norm(R_target2cam[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean += error

    print(f"Reprojection error: {mean / len(R_gripper2base)}")
    return transformation_dict


if __name__ == "__main__":
    # folders = ["calibration0", "calibration-90_2"]
    folders = ["calibration0"]
    for folder in folders:
        print(f"Processing: {folder}")
        # folder = "calibration0"
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        transf_d = extrensic_calibration(folder)
        if transf_d is None:
            print("Not enough images")
            continue
        with open(
            os.path.join(folder, "cam2flange_transf_" + folder + ".json"), "w"
        ) as f:
            json.dump(transf_d, f, indent=2)
