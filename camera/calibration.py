import os
import argparse
import json

import numpy as np
import cv2
from tqdm import tqdm

POSSIBLE_VERTICIES = [
    (6, 8),
    (5,8)
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path2images",
        type=str,
        help="Path to folder containing images for calibration",
        nargs="+",
    )
    parser.add_argument(
        "--checker_box_size",
        type=str,
        help="Number of inner vortexes in the checkerbox",
        default="(7, 8)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save camera parameters",
        default="camera_parameters.json",
    )
    parser.add_argument(
        "--vertices",
        type=int,
        help="Number of vertices in the calibration pattern should be 2 numbers",
        nargs=2,
        required=True,
    )
    args = parser.parse_args()
    #
    args.vertices = tuple(args.vertices)

    if len(args.path2images) == 1:
        args.path2images = args.path2images[0]

    return args


def draw(image, corners, image_points):
    corner = tuple(corners[0].ravel().astype(int))

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


def get_camera_parameters_from_images_multiple(
    path2images: list, checker_box_size: tuple = (6, 8)
) -> dict:
    object_points = []  # 3d point in real world space
    image_points = []  # 2d points in image plane
    mean = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    for img_path in path2images:
        print(f"Processing {img_path}")
        image_names = sorted(os.listdir(img_path))
        checker_idx = 0
        idx_found = False
        for image_name in tqdm(image_names):
            image = cv2.imread(os.path.join(img_path, image_name))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret = False
            for e, checkers_size in enumerate(POSSIBLE_VERTICIES):
                checker_box_size = POSSIBLE_VERTICIES[e]
                ret, corners = cv2.findChessboardCorners(gray, checker_box_size, None)
                checker_idx = e

                if ret:
                    idx_found = True
                    cv2.drawChessboardCorners(image, checker_box_size, corners, ret)
                    cv2.imshow("img", image)
                    cv2.waitKey(0)

                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    object_points.append(
                        np.zeros((checker_box_size[0] * checker_box_size[1], 3), np.float32)
                    )
                    object_points[-1][:, :2] = np.mgrid[
                        0 : checker_box_size[0], 0 : checker_box_size[1]
                    ].T.reshape(-1, 2)
                    image_points.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None
    )
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], mtx, dist
        )
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean += error

    camera_parameters = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "reprojection_error": mean / len(object_points),
        "height": gray.shape[0],
        "width": gray.shape[1],
    }
    return camera_parameters

def get_camera_parameters_from_images(
    path2images: str, checker_box_size: tuple = (6, 8)
) -> dict:
    """Get camera parameters from images

    Args:
        path2images (str): Path to folder containing images
        checker_size (tuple, optional): NUmber of inner vortexes in the checkerbox. Defaults to (7, 8).

    Returns:
        dict: Camera parameters, including camera matrix, distortion coefficients, rotation and translation vectors
    """
    image_names = sorted(os.listdir(path2images))
    object_points = []  # 3d point in real world space
    image_points = []  # 2d points in image plane
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    for image_name in tqdm(image_names):
        image = cv2.imread(os.path.join(path2images, image_name))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checker_box_size, None)

        cv2.drawChessboardCorners(image, checker_box_size, corners, ret)
        cv2.imshow("img", image)
        cv2.waitKey(0)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(
                np.zeros((checker_box_size[0] * checker_box_size[1], 3), np.float32)
            )
            object_points[-1][:, :2] = np.mgrid[
                0 : checker_box_size[0], 0 : checker_box_size[1]
            ].T.reshape(-1, 2)
            image_points.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None
    )

    mean_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], mtx, dist
        )
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((checker_box_size[0] * checker_box_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : checker_box_size[1], 0 : checker_box_size[0]].T.reshape(
        -1, 2
    )
    axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)

    for e, image_name in enumerate(tqdm(image_names)):
        image = cv2.imread(os.path.join(path2images, image_name))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, checker_box_size, None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            print(rvecs[e], tvecs[e])
            # Find the rotation and translation vectors.
            ret, rvecs2, tvecs2 = cv2.solvePnP(objp, corners2, mtx, dist)
            print(rvecs2, tvecs2)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs[e], tvecs[e], mtx, dist)

            image = draw(image, corners2, imgpts)
            cv2.imshow("img", image)
            cv2.waitKey(0)

    camera_parameters = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "reprojection_error": mean_error / len(object_points),
        "height": gray.shape[0],
        "width": gray.shape[1],
    }

    cv2.destroyAllWindows()
    return camera_parameters


if __name__ == "__main__":
    args = parse_args()
    # camera_params = get_camera_parameters_from_images(args.path2images)
    # with open(args.save_path, "w") as f:
    #     json.dump(camera_params, f, indent=2)
