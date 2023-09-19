import os
import argparse
import json

import numpy as np
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path2images",
        type=str,
        help="Path to folder containing images for calibration",
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
    args = parser.parse_args()
    return args


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
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    for image_name in tqdm(image_names):
        image = cv2.imread(os.path.join(path2images, image_name))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checker_box_size, None)

        # cv2.drawChessboardCorners(gray, checker_box_size, corners, ret)
        # cv2.imshow("img", gray)
        # cv2.waitKey(0)

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

    for e, image_name in enumerate(tqdm(image_names)):
        image = cv2.imread(os.path.join(path2images, image_name))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        # visualize rvects and tvects
        imgpts, jac = cv2.projectPoints(
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
            rvecs[e],
            tvecs[e],
            mtx,
            dist,
        )
        image = cv2.line(image, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 255, 0), 3)
        imgpts, jac = cv2.projectPoints(
            np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]),
            rvecs[e],
            tvecs[e],
            mtx,
            dist,
        )
        image = cv2.line(image, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (255, 0, 0), 3)
        imgpts, jac = cv2.projectPoints(
            np.array([(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
            rvecs[e],
            tvecs[e],
            mtx,
            dist,
        )
        image = cv2.line(image, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 3)
        cv2.imshow("img", image)
        cv2.waitKey(0)



    camera_parameters = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "reprojection_error": mean_error / len(object_points),
        "height": gray.shape[0],
        "width": gray.shape[1],
    }
    return camera_parameters


if __name__ == "__main__":
    args = parse_args()
    camera_params = get_camera_parameters_from_images(args.path2images)
    with open(args.save_path, "w") as f:
        json.dump(camera_params, f, indent=2)
