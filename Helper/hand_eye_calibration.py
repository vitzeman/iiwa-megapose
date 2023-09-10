import math
import os.path

import cv2
import numpy as np
# import matplotlib.pyplot as plt
#
# import glob
# import random
# import sys
import json
from rotation_helper import rotation_angles

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
# Note: Pattern generated using the following link
# https://calib.io/pages/camera-calibration-pattern-generator
board = cv2.aruco.CharucoBoard_create(11, 8, 0.014, 0.012, aruco_dict)


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

def get_RT(frame, camera_matrix, dist_coeff, board, distortion_coefficients ):
    corners, ids, rejected_points = cv2.aruco.detectMarkers(frame, aruco_dict)
    tv = np.array([0,0,0])
    rv = np.zeros((3,3))
    # print(len(ids), len(corners), len(rejected_points))

    if corners is None or ids is None:
        return None, None
    if len(corners) != len(ids) or len(corners) == 0:
        return None, None

    ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners,
                                                                ids,
                                                                frame,
                                                                board)





    ret, p_rvec, p_tvec = cv2.aruco.estimatePoseCharucoBoard(c_corners,
                                                            c_ids,
                                                            board,
                                                            camera_matrix,
                                                            dist_coeff,
                                                            np.empty(1),
                                                            np.empty(1))


    if p_rvec is None or p_tvec is None:
        return None, None
    if np.isnan(p_rvec).any() or np.isnan(p_tvec).any():
        return None, None

    return p_rvec, p_tvec



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
                                                                dist_coeff,
                                                                np.empty(1),
                                                                np.empty(1))
        if p_rvec is None or p_tvec is None:
            return None
        if np.isnan(p_rvec).any() or np.isnan(p_tvec).any():
            return None
        # cv2.aruco.drawAxis(frame,
        #                 camera_matrix,
        #                 dist_coeff,
        #                 p_rvec,
        #                 p_tvec,
        #                 0.1)
        cv2.aruco.drawDetectedCornersCharuco(frame, c_corners, c_ids)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.aruco.drawDetectedMarkers(frame, rejected_points, borderColor=(100, 0, 240))


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

def get_base_pose(dir):

    head, tail = os.path.split(os.path.split(dir)[0])
    pose_dir = os.path.join(head,tail, 'pose')
    pose_file = open(os.path.join(pose_dir, 'pose.json'))
    pose_json = json.load(pose_file)

    return pose_json

def get_pose_dir(dir):
    head, tail = os.path.split(os.path.split(dir)[0])
    pose_dir = os.path.join(head,tail, 'pose')
    return pose_dir

def find_transform(rotation, tran):
    T = np.zeros((4, 4))
    R = rotation
    T[0,3] = (tran[0])
    T[1,3] = (tran[1])
    T[2,3] = (tran[2])
    T[3,3] = 1
    T[:3,:3] = R

    return T

def save_marker_pose(frame, transf, camera_matrix, out_dict):

    out_dict[frame] = {}
    out_dict[frame]['W2C'] = transf.tolist()
    out_dict[frame]['K'] = camera_matrix.tolist()
    out_dict[frame]['img_size'] = [2448, 2048]

def print_things(key, angle_marker, Tvec, angle_flange, tran_flange):
    print(key)
    print("Marker transformations")

    print(math.degrees(angle_marker[0]) % 360, math.degrees(angle_marker[1]) % 360, math.degrees(angle_marker[2]) % 360)
    print(Tvec)
    print("  ")

    print("Flange transformation")

    print(math.degrees(angle_flange[0]) % 360, math.degrees(angle_flange[1]) % 360, math.degrees(angle_flange[2]) % 360)
    print(tran_flange)
    # rot_rodri, _ = cv2.Rodrigues(rot_flange)
    # print(rot_rodri)
    print("  ")

def main():
    out_dict = {}
    head, tail = os.path.split(os.path.split(os.getcwd())[0])
    img_dir = os.path.join(head,tail, 'dataset', 'single_correct_charuco2', 'images')
    pose = get_base_pose(img_dir)

    camera_matrix = np.array([[2937, 0,      1224],
                       [0,      2937, 1024],
                       [0,      0,          1]])
    dist_coeff = np.array([0, 0, 0, 0])

    # camera_matrix = np.array([[1.78421785e+02, 0.00000000e+00, 1.01997916e+03],
    #                          [0.00000000e+00, 2.68036434e+02, 1.22139280e+03],
    #                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    #
    # dist_coeff = np.array([-3.72607971e-03, -1.50000843e-04, -4.27030872e-03,  2.79202734e-03,  -6.12641910e-06])

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for key in pose:
        img_path = os.path.join(img_dir, key + '.jpg')
        frame = cv2.imread(img_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        trans_flange = np.array(pose[key]['W2C'])
        rot_flange = trans_flange[:3,:3]
        tran_flange = trans_flange[:3,3:]

        Rvec, Tvec = get_RT(frame= gray, camera_matrix= camera_matrix, dist_coeff= dist_coeff, board=board, distortion_coefficients=dist_coeff)

        if len(Rvec)<2:
            continue

        print(Rvec)
        print(Tvec)

        # Rvector = np.array([Rvec[1], Rvec[2], Rvec[0]])

        rot_marker,_ = cv2.Rodrigues(Rvec)

        angle_marker = rotation_angles(rot_marker, 'zyx')
        angle_flange = rotation_angles(rot_flange, 'zyx')

        marker_tranf = find_transform(rot_marker, Tvec)

        print_things(key, angle_marker, Tvec, angle_flange, tran_flange)
        # save_marker_pose(key + '.jpg', marker_tranf, camera_matrix, out_dict)


        R_gripper2base.append(rot_flange)
        t_gripper2base.append(tran_flange)
        R_target2cam.append(rot_marker)
        t_target2cam.append(Tvec)

    R_gripper2base = np.array(R_gripper2base)
    t_gripper2base = np.array(t_gripper2base)
    R_target2cam = np.array(R_target2cam)
    t_target2cam = np.array(t_target2cam)

    # print(t_target2cam[:,1])
    # print((t_gripper2base[:,1]))

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base=R_gripper2base, t_gripper2base=t_gripper2base,
                                                        R_target2cam=R_target2cam, t_target2cam=t_target2cam,
                                                        method= cv2.CALIB_HAND_EYE_TSAI)

    print(R_cam2gripper)
    print(t_cam2gripper)

    angle = rotation_angles(R_cam2gripper, 'zyx')
    print(math.degrees(angle[0]), math.degrees(angle[1]),math.degrees( angle[2]) )

    # pose_dir = get_pose_dir(img_dir)
    # with open(os.path.join(pose_dir, "marker_pose.json"), "w") as outfile:
    #     json.dump(out_dict, outfile)

if __name__ == '__main__':
    main()