import numpy as np
import cv2 as cv
import glob
import os
import json
import math
from rotation_helper import rotation_angles, rotation_matrix

class calibration:

    def __init__(self):
        pass

    def intrinsics(self):
        rows = 8
        columns = 7
        sizeofsquare = 0.2
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((columns*rows,3), np.float32)
        objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        head, tail = os.path.split(os.path.split(os.getcwd())[0])
        img_dir = os.path.join(head, tail, 'dataset', 'chessboard', 'images')
        camera_mtx = np.array([[0.0, 0.0, 0],
                               [0.0, 0.0, 0],
                               [0.0, 0.0, 0]])

        dist_mtx = np.array([[0, 0, 0, 0, 0]])

        images = glob.glob(os.path.join(img_dir,'*.jpg'))
        for fname in images:
            print(fname)
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (columns,rows), None)
            cv.drawChessboardCorners(img, (columns, rows), corners, ret)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners, (5,5), (-1,-1), criteria)
                imgpoints.append(corners2)

        # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_mtx, dist_mtx)
        ret, mtx, dist, rvecs, tvecs =  cv.calibrateCamera(objectPoints=objpoints, imagePoints=imgpoints,
                                    imageSize= gray.shape[::-1],cameraMatrix=camera_mtx, distCoeffs=dist_mtx,)
                                    # flags= cv.CALIB_FIX_PRINCIPAL_POINT +  cv.CALIB_SAME_FOCAL_LENGTH )
        height, width, depth = img.shape

        print(mtx)
        print(dist)

        ## reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print( "total error: {}".format(mean_error/len(objpoints)) )

        config_dir = os.path.join(head, tail, 'config')
        out_dict = {}
        out_dict['camera_matrix'] = mtx.tolist()
        out_dict['dist_coeff'] = dict()
        out_dict['dist_coeff']['k1'] = dist[0][0]
        out_dict['dist_coeff']['k2'] = dist[0][1]
        out_dict['dist_coeff']['p1'] = dist[0][2]
        out_dict['dist_coeff']['p2'] = dist[0][3]
        out_dict['dist_coeff']['k3'] = dist[0][4]

        out_dict['img_size'] = [width, height]

        with open(os.path.join(config_dir, "camera.json"), "w") as outfile:
            json.dump(out_dict, outfile, indent= 2)


    def find_transform(self, rotation, tran):
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

    def print_things(self,key, robot_trans, marker_trans):
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

    def make_dist_matrix(self,dist):
        dist_mtx = np.array([])
        for key in dist:
            dist_mtx = np.append(dist_mtx,dist[key])
        print(dist_mtx)

    def get_base_pose(self,dir):

        head, tail = os.path.split(os.path.split(dir)[0])
        pose_dir = os.path.join(head,tail, 'pose')
        pose_file = open(os.path.join(pose_dir, 'pose.json'))
        pose_json = json.load(pose_file)

        return pose_json

    def get_pose_dir(self,dir):
        head, tail = os.path.split(os.path.split(dir)[0])
        pose_dir = os.path.join(head,tail, 'pose')
        return pose_dir


    def extrinsics(self):

        rows = 8
        columns = 7
        sizeofsquare = 0.2
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((columns * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        out_dict = {}

        head, tail = os.path.split(os.path.split(os.getcwd())[0])
        config_dir = os.path.join(head, tail, 'config')
        img_dir = os.path.join(head, tail, 'dataset', 'chessboard', 'images')
        pose = self.get_base_pose(img_dir)
        camera_file = open(os.path.join(config_dir, 'camera.json'))
        camera = json.load(camera_file)
        camera_matrix = np.array(camera['camera_matrix'])
        dist_coeff = self.make_dist_matrix(camera['dist_coeff'])

        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for key in pose:
            img_path = os.path.join(img_dir, key)
            frame = cv.imread(img_path)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            trans_flange_w2c = np.array(pose[key]['W2C'])
            rot_flange_w2c = trans_flange_w2c[:3, :3]
            tran_flange_w2c = trans_flange_w2c[:3, 3:]

            trans_flange_c2w = np.linalg.inv(trans_flange_w2c)

            rot_flange_c2w = trans_flange_c2w[:3, :3]
            tran_flange_c2w = trans_flange_c2w[:3, 3:] * 0.001

            ret, corners = cv.findChessboardCorners(frame, (columns, rows), None)
            cv.drawChessboardCorners(frame, (columns, rows), corners, ret)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(frame, corners, (5, 5), (-1, -1), criteria)
                imgpoints.append(corners2)




            ret, mtx, dist, Rvec, Tvec = cv.calibrateCamera(objectPoints=objpoints, imagePoints=imgpoints,
                                                              imageSize=frame.shape[::-1], cameraMatrix=mtx,
                                                              distCoeffs=dist_coeff)

                rot_marker, _ = cv.Rodrigues(Rvec)

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




    R_cam2gripper, t_cam2gripper = cv.calibrateHandEye(R_gripper2base=R_gripper2base, t_gripper2base=t_gripper2base,
                                                        R_target2cam=R_target2cam, t_target2cam=t_target2cam,
                                                        method=cv2.CALIB_HAND_EYE_TSAI)

    print("gripper to cam rotational error")
    print(R_cam2gripper)

    print(" gripper to cam translation error in mm")
    print(t_cam2gripper * 1000)

    print("gripper to cam rotation error in degree")
    angle = rotation_angles(R_cam2gripper, 'zyx')
    print(angle[0], angle[1], angle[2])


