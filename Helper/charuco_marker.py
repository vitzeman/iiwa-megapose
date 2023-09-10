import cv2
import numpy as np
import matplotlib.pyplot as plt

import glob
import random
import sys
import os

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
# Note: Pattern generated using the following link
# https://calib.io/pages/camera-calibration-pattern-generator
board = cv2.aruco.CharucoBoard_create(11, 8, 0.015, 0.011, aruco_dict)


def read_chessboards(frames):
    """
    Charuco base pose estimation.
    """
    all_corners = []
    all_ids = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners) > 0:
            ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            # ret is the number of detected corners
            if ret > 0:
                all_corners.append(c_corners)
                all_ids.append(c_ids)
        else:
            print('Failed!')

    imsize = gray.shape
    return all_corners, all_ids, imsize


def capture_camera(dev_num=0, num=1, mirror=False, size=None):
    frames = []

    cap = cv2.VideoCapture(dev_num)

    while True:
        ret, frame = cap.read()

        if mirror is True:
            frame = cv2.flip(frame, 1)

        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        # My config applies floating layout for windows named 'Java'
        cv2.imshow('Java', frame)

        k = cv2.waitKey(1)
        if k == 27:  # Esc
            break
        elif k == 10 or k == 32:  # Enter or Space
            frames.append(frame)
            print('Frame captured!')
            if len(frames) == num:
                break

    return frames


def draw_axis(frame, camera_matrix, dist_coeff, board, verbose=True):
    corners, ids, rejected_points = cv2.aruco.detectMarkers(frame, aruco_dict)
    if corners is None or ids is None:
        return None
    if len(corners) != len(ids) or len(corners) == 0:
        return None

    try:
        ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners,
                                                                    ids,
                                                                    frame,
                                                                    board)
        ret, p_rvec, p_tvec = cv2.aruco.estimatePoseCharucoBoard(c_corners,
                                                                c_ids,
                                                                board,
                                                                camera_matrix,
                                                                dist_coeff)
        if p_rvec is None or p_tvec is None:
            return None
        if np.isnan(p_rvec).any() or np.isnan(p_tvec).any():
            return None
        cv2.aruco.drawAxis(frame,
                        camera_matrix,
                        dist_coeff,
                        p_rvec,
                        p_tvec,
                        0.1)
        # cv2.aruco.drawDetectedCornersCharuco(frame, c_corners, c_ids)
        # cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # cv2.aruco.drawDetectedMarkers(frame, rejected_points, borderColor=(100, 0, 240))
    except cv2.error:
        return None

    if verbose:
        print('Translation : {0}'.format(p_tvec))
        print('Rotation    : {0}'.format(p_rvec))
        print('Distance from camera: {0} m'.format(np.linalg.norm(p_tvec)))

    return frame


def get_frame(img_dir):
    frames = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir,img_name)
        img = cv2.imread(img_path)
        height, width, none = img.shape
        frames.append(img)

    return frames, height,width

def save_undistorted(path, frames, camera_matrix, dist_coeffs, newcameramatrix):
     head, tail = os.path.split(os.path.split(path)[0])
     undis_dir = os.path.join(head,'undist')
     if not os.path.isdir(undis_dir):
         os.mkdir(undis_dir)
     i = 0
     for frame in frames:
         undistorted_image = cv2.undistort(
             frame, camera_matrix, dist_coeffs, None, newcameramatrix
         )
         # cv2.imshow("undistorted", undistorted_image)
         img_loc = os.path.join(undis_dir, str(i)+'.jpeg')
         cv2.imwrite(img_loc,undistorted_image )
         i = i+1

def main():
    head, tail = os.path.split(os.path.split(os.getcwd())[0])
    img_dir = os.path.join(head, tail, 'dataset', 'charuco1', 'images')
    frames, height, width = get_frame(img_dir)
    all_corners, all_ids, imsize = read_chessboards(frames)
    all_corners = [x for x in all_corners if len(x) >= 4]
    all_ids = [x for x in all_ids if len(x) >= 4]
    ret, camera_matrix, dist_coeff, rvec, tvec = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, imsize, None, None
    )

    print('> Camera matrix')
    print(camera_matrix)
    print('> Distortion coefficients')
    print(dist_coeff)
    for frame in frames:
        axis_frame = draw_axis(frame, camera_matrix, dist_coeff, board, True)
        print(axis_frame)




if __name__ == '__main__':
    main()