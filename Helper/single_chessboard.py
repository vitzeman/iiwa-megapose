import numpy as np
import cv2
from pypylon import pylon
from rotation_helper import rotation_angles, rotation_matrix
import math
import os
import json

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2) * 0.024
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)


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


def save_marker_pose(frame, transf, camera_matrix, out_dict):

    out_dict[frame] = {}
    out_dict[frame]['W2C'] = transf.tolist()
    out_dict[frame]['K'] = camera_matrix.tolist()
    out_dict[frame]['img_size'] = [2448, 2048]

def print_things(key, angle_marker, Tvec, angle_flange, tran_flange):
    print(key)
    print("Marker transformations")

    print(math.degrees(angle_marker[0]) % 360, math.degrees(angle_marker[1]) % 360, math.degrees(angle_marker[2]) % 360)
    Tvec = Tvec.reshape(-1,1)
    print(Tvec)
    print("  ")

    print("Flange transformation")

    print(math.degrees(angle_flange[0]) % 360, math.degrees(angle_flange[1]) % 360, math.degrees(angle_flange[2]) % 360)
    print(tran_flange)
    print("  ")

def load_camera():
    root = os.getcwd()
    config_dir = os.path.join(root, 'config')
    camera_file = open(os.path.join(config_dir, 'camera.json'))
    camera = json.load(camera_file)
    return camera

def main():
    out_dict = {}

    head, tail = os.path.split(os.path.split(os.getcwd())[0])
    config_dir = os.path.join(head,tail, 'config')
    img_dir = os.path.join(head, tail, 'dataset', 'chessboard_1', 'images')
    pose = get_base_pose(img_dir)
    camera_file = open(os.path.join(config_dir, 'camera.json'))
    camera = json.load(camera_file)
    camera_matrix = np.array(camera['camera_matrix'])
    dist_coeff = np.array(camera['dist_coeff'])

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for key in pose:
        img_path = os.path.join(img_dir, key)
        frame = cv2.imread(img_path)
        trans_flange = np.array(pose[key]['W2C'])

        trans_flange = np.linalg.inv(trans_flange)

        rot_flange = trans_flange[:3, :3]
        tran_flange = trans_flange[:3, 3:]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (10, 7), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeff)


        marA = math.degrees(rvecs[2])
        marB = math.degrees(rvecs[1])
        marC = math.degrees(rvecs[0])

        arX = tvecs[0]
        arY = tvecs[1]
        arZ = tvecs[2]


        rot_marker = rotation_matrix(marA, marB, marC, order='zyx')

        Tm = np.zeros((4, 4))
        Tm[0, 3] = arX
        Tm[1, 3] = arY
        Tm[2, 3] = arZ
        Tm[:3, :3] = rot_marker
        Tm[3, 3] = 1
        rot_marker = Tm[:3, :3]
        Tran_marker = Tm[:3, 3]

        # ##### c2W
        # Tm = np.linalg.inv(Tm)
        # rot_marker = Tm[:3, :3]
        # Tran_marker = Tm[:3, 3]

        # angle_marker = rotation_angles(rot_marker, 'zyx') #zxy
        # angle_flange = rotation_angles(rot_flange, 'zyx')
        # marker_tranf = find_transform(rot_marker, Tvec[0][0])
        # marker_matrix_ngp = generate_transform_matrix(Tvec[0][0], rot_marker)
        # marker_tranf = np.linalg.inv(marker_tranf)
        # print_things(key, angle_marker, Tvec, angle_flange, tran_flange)
        save_marker_pose(key, Tm, camera_matrix, out_dict)

        R_gripper2base.append(rot_flange)
        t_gripper2base.append(tran_flange)
        R_target2cam.append(rot_marker)
        t_target2cam.append(Tran_marker)

    R_gripper2base = np.array(R_gripper2base)
    t_gripper2base = np.array(t_gripper2base)
    R_target2cam = np.array(R_target2cam)
    t_target2cam = np.array(t_target2cam)


    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base=R_gripper2base, t_gripper2base=t_gripper2base,
                                                        R_target2cam=R_target2cam, t_target2cam=t_target2cam,
                                                        method= cv2.CALIB_HAND_EYE_TSAI)

    print("gripper to cam translation and rotational error")
    print(R_cam2gripper)
    print(t_cam2gripper * 1000 )
    # R_cam2gripper = np.linalg.inv(R_cam2gripper)

    angle = rotation_angles(R_cam2gripper, 'zyx')
    print(angle[0], angle[1], angle[2])

    pose_dir = get_pose_dir(img_dir)
    with open(os.path.join(pose_dir, "marker_pose.json"), "w") as outfile:
        json.dump(out_dict, outfile)


if __name__ == '__main__':
    main()
