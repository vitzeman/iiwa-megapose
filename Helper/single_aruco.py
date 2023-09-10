import numpy as np
import cv2
import os
import json
from rotation_helper import rotation_angles, rotation_matrix
import math

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
marker_size = 0.175

def pose_esitmation(frame, aruco_dict, matrix_coefficients, distortion_coefficients):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict)

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, matrix_coefficients,
                                                                           distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            # cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return rvec, tvec

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

def generate_transform_matrix(pos, rot):
    # tran = np.eye(4)
    # tran[:3,:3] = rot
    # tran[:3, 3] = pos
    #
    # # tran = np.linalg.inv(tran)
    # rot = tran[:3,:3]
    # pos = tran[:3, 3] * 13

    xf_rot = np.eye(4)
    xf_rot[:3,:3] = rot

    xf_pos = np.eye(4)
    xf_pos[:3,3] = pos * 4

    # barbershop_mirros_hd_dense:
    # - camera plane is y+z plane, meaning: constant x-values
    # - cameras look to +x
    # Don't ask me...
    extra_xf = np.matrix([
        [-1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]])
    # NerF will cycle forward, so lets cycle backward.
    shift_coords = np.matrix([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])
    xf = shift_coords @ extra_xf @ xf_pos
    assert np.abs(np.linalg.det(xf) - 1.0) < 1e-4
    xf = xf @ xf_rot
    xf = np.linalg.inv(xf)
    return xf

def find_transform(rotation, tran):
    T = np.zeros((4, 4))
    R = rotation
    T[0,3] = (tran[0])
    T[1,3] = (tran[1])
    T[2,3] = (tran[2])
    T[3,3] = 1
    T[:3,:3] = R
    T = np.linalg.inv(T)

    return T

def save_marker_pose(frame, transf, camera_matrix, out_dict):

    out_dict[frame] = {}
    out_dict[frame]['K'] = camera_matrix.tolist()
    out_dict[frame]['img_size'] = [2448, 2048]
    out_dict[frame]['W2C'] = transf.tolist()

def print_things(key, robot_trans, marker_trans):
    print(key)

    print("Marker transformations")
    Tvec_marker =  marker_trans[:3, 3:]
    Rot_marker = marker_trans[:3, :3]
    angle_marker = rotation_angles(Rot_marker, 'zyx')
    print(math.degrees(angle_marker[0])%360, math.degrees(angle_marker[1])%360, math.degrees(angle_marker[2])%360)
    print(Tvec_marker)
    print("  ")

    print("Flange transformation")
    Tvec_robot = robot_trans[:3, 3:]
    Rot_robot = robot_trans[:3, :3] * 0.001
    angle_robot = rotation_angles(Rot_robot, 'zyx')
    print(math.degrees(angle_robot[0])%360, math.degrees(angle_robot[1])%360, math.degrees(angle_robot[2])%360)

    print(Tvec_robot)
    print("  ")

def make_dist_matrix(dist):
    dist_mtx = np.array([])
    for key in dist:
        dist_mtx = np.append(dist_mtx,dist[key])
    print(dist_mtx)

def main():
    out_dict = {}

    head, tail = os.path.split(os.path.split(os.getcwd())[0])
    config_dir = os.path.join(head,tail, 'config')
    img_dir = os.path.join(head, tail, 'dataset', 'marker', 'images')
    pose = get_base_pose(img_dir)
    camera_file = open(os.path.join(config_dir, 'camera.json'))
    camera = json.load(camera_file)
    camera_matrix = np.array(camera['camera_matrix'])
    dist_coeff = make_dist_matrix(camera['dist_coeff'])

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for key in pose:
        img_path = os.path.join(img_dir, key)
        frame = cv2.imread(img_path)
        trans_flange_w2c = np.array(pose[key]['W2C'])
        rot_flange_w2c = trans_flange_w2c[:3, :3]
        tran_flange_w2c = trans_flange_w2c[:3, 3:]

        trans_flange_c2w = np.linalg.inv(trans_flange_w2c)

        rot_flange_c2w = trans_flange_c2w[:3, :3]
        tran_flange_c2w = trans_flange_c2w[:3, 3:] * 0.001

        try:

            Rvec, Tvec = pose_esitmation(frame=frame, aruco_dict= aruco_dict, matrix_coefficients= camera_matrix, distortion_coefficients= dist_coeff)

            rot_marker, _ = cv2.Rodrigues(Rvec)

            angle_marker_w2c = rotation_angles(rot_marker, 'zyx')  # zxy
            w2c_marA = round(angle_marker_w2c[0], 5)
            w2c_marB = round(angle_marker_w2c[1], 5)
            w2c_marC = round(angle_marker_w2c[2], 5)

            w2c_arX = round(Tvec[0][0][0], 5)
            w2c_arY = round(Tvec[0][0][1], 5)
            w2c_arZ = round(Tvec[0][0][2], 5)

            rot_marker_w2c = rotation_matrix(w2c_marA, w2c_marB, w2c_marC, order='zyx')
            Tm_w2c = np.zeros((4, 4))
            Tm_w2c[0, 3] = w2c_arX
            Tm_w2c[1, 3] = w2c_arY
            Tm_w2c[2, 3] = w2c_arZ
            Tm_w2c[:3, :3] = rot_marker_w2c
            Tm_w2c[3, 3] = 1

            Tm_w2c_rot = Tm_w2c[:3, :3]
            Tm_w2c_tran = Tm_w2c[:3, 3:]

            # ##### c2W
            Tm_c2w = np.linalg.inv(Tm_w2c)

            rot_marker_c2w = Tm_c2w[:3, :3]
            c2w_arX = round(Tm_c2w[0, 3], 5)
            c2w_arY = round(Tm_c2w[1, 3], 5)
            c2w_arZ = round(Tm_c2w[2, 3], 5)
            angle_marker_c2w = rotation_angles(rot_marker_c2w, 'zyx')  # zxy
            c2w_marA = round(angle_marker_c2w[0], 5)
            c2w_marB = round(angle_marker_c2w[1], 5)
            c2w_marC = round(angle_marker_c2w[2], 5)

            Tm_c2w_rot = Tm_c2w[:3, :3]
            Tm_c2w_tran = Tm_c2w[:3, 3:]

            # angle_marker = rotation_angles(rot_marker, 'zyx') #zxy
            # angle_flange = rotation_angles(rot_flange, 'zyx')
            # marker_tranf = find_transform(rot_marker, Tvec[0][0])
            # marker_matrix_ngp = generate_transform_matrix(Tvec[0][0], rot_marker)
            # marker_tranf = np.linalg.inv(marker_tranf)
            print_things(key,trans_flange_w2c, Tm_w2c)

            save_marker_pose(key, Tm_w2c, camera_matrix, out_dict)

            R_gripper2base.append(rot_flange_c2w)
            t_gripper2base.append(tran_flange_c2w)
            R_target2cam.append(Tm_w2c_rot)
            t_target2cam.append(Tm_w2c_tran)
        except:
            print("couldnt find transformation of frame", key)

    R_gripper2base = np.array(R_gripper2base)
    t_gripper2base = np.array(t_gripper2base)
    R_target2cam = np.array(R_target2cam)
    t_target2cam = np.array(t_target2cam)

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base=R_gripper2base, t_gripper2base=t_gripper2base,
                                                        R_target2cam=R_target2cam, t_target2cam=t_target2cam,
                                                        method= cv2.CALIB_HAND_EYE_TSAI)

    print("gripper to cam rotational error")
    print(R_cam2gripper)

    print(" gripper to cam translation error in mm")
    print(t_cam2gripper * 1000)

    print("sum of error in mm")
    print(np.sum(t_cam2gripper * 1000))


    print("gripper to cam rotation error in degree")
    angle = rotation_angles(R_cam2gripper, 'zyx')
    print(angle[0], angle[1], angle[2])

    print("sum of error in angle")
    print(np.sum(angle))


    pose_dir = get_pose_dir(img_dir)
    with open(os.path.join(pose_dir, "marker_pose.json"), "w") as outfile:
        json.dump(out_dict, outfile, indent=2)


if __name__ == '__main__':
    main()
