from opcua import Client
import numpy as np
import cv2
from pypylon import pylon
from utils.robot_motion import get_angle_and_translation
from rotation_helper import rotation_angles, rotation_matrix
import math
import os
import json

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

## Set things to auto for best image possible
camera.GainAuto.SetValue("Once")
camera.ExposureAuto.SetValue("Once")
camera.BalanceWhiteAuto.SetValue("Once")

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
marker_size = 0.174

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
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.15)

    return rvec, tvec

def Rz(theta):
  return np.matrix([[ math.cos(math.radians(theta)), -math.sin(math.radians(theta)), 0 ],
                   [ math.sin(math.radians(theta)), math.cos(math.radians(theta)) , 0 ],
                   [ 0           , 0            , 1 ]])


def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, math.cos(theta), -math.sin(theta)],
                      [0, math.sin(theta), math.cos(theta)]])


def Ry(theta):
    return np.matrix([[math.cos(theta), 0, math.sin(theta)],
                      [0, 1, 0],
                      [-math.sin(theta), 0, math.cos(theta)]])

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def converPoseNGP(c2w):
    c2w[0:3, 2] *= -1  # flip the y and z axis
    c2w[0:3, 1] *= -1
    c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
    c2w[2, :] *= -1  # flip whole world upside down

    return c2w

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

flip_mat = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, -1, 0],
			[0, 0, 0, 1]
		])

def convertposewithflip(C2W):
    return  flip_mat @ C2W @ flip_mat

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def make_dist_matrix(dist):
    dist_mtx = np.array([])
    for key in dist:
        dist_mtx = np.append(dist_mtx,dist[key])
    return dist_mtx

def main():
    client = Client("opc.tcp://localhost:5000/")
    try:

        client.connect()
        client.load_type_definitions()
        root = client.get_root_node()
        objects = client.get_objects_node()
        uri = "http://iiwa-control.ciirc.cz"
        idx = client.get_namespace_index(uri)
        rootObj = objects.get_child(["IIWA"])
        input = rootObj.get_child(["RobotGenericInput"])
        output = rootObj.get_child(["RobotGenericOutput"])
        print("connection started")

        # camera_matrix = np.array([[2.30090784e+03, 0.00000000e+00, 1.21819207e+03],
        #                           [0.00000000e+00, 2.29285500e+03, 9.91465866e+02],
        #                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # dist_coeff = np.array([-0.03214205, - 0.17672478, 0.00181945, - 0.00413327, 0.70466039])


        # camera_matrix = np.array([[2.30090784e+03, 0.00000000e+00, 1224],
        #                           [0.00000000e+00, 2.29285500e+03, 1024],
        #                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # dist_coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # point_num = 0
        head, tail = os.path.split(os.path.split(os.getcwd())[0])
        config_dir = os.path.join(head, tail, 'config')
        camera_file = open(os.path.join(config_dir, 'camera.json'))
        camera_dict = json.load(camera_file)
        camera_matrix = np.array(camera_dict['camera_matrix'])
        dist_coeff = make_dist_matrix(camera_dict['dist_coeff'])

        while camera.IsGrabbing():
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                image = image.GetArray()
                height, width, channel = image.shape

                RX, RY, RZ, RA, RB, RC = get_angle_and_translation(input=input)

                c2w_RX = round(RX * 0.001 , 5)
                c2w_RY = round(RY * 0.001 , 5)
                c2w_RZ = round(RZ * 0.001 , 5)
                c2w_RA = round(RA, 5)
                c2w_RB = round(RB, 5)
                c2w_RC = round(RC, 5)

                rot_rob_c2w = rotation_matrix(c2w_RA, c2w_RB, c2w_RC, order='zyx')
                Tc2w = np.zeros((4, 4))
                Tc2w[0, 3] = c2w_RX
                Tc2w[1, 3] = c2w_RY
                Tc2w[2, 3] = c2w_RZ
                Tc2w[:3, :3] = rot_rob_c2w
                Tc2w[3, 3] = 1

                ##### w2C
                Tw2c = np.linalg.inv(Tc2w)

                rot_rob = Tw2c[:3, :3]
                w2c_RX = round(Tw2c[0, 3], 5)
                w2c_RY = round(Tw2c[1, 3], 5)
                w2c_RZ = round(Tw2c[2, 3], 5)
                angle_rob_w2c = rotation_angles(rot_rob, 'zyx')  # zxy
                w2c_RA = round(angle_rob_w2c[0], 5)
                w2c_RB = round(angle_rob_w2c[1], 5)
                w2c_RC = round(angle_rob_w2c[2], 5)

                robot_str_c2w = 'robot c2w  ' + '   X  ' + str(c2w_RX) + '   Y  ' + str(c2w_RY) + '  Z   ' + str(c2w_RZ) + '   A   ' + str(c2w_RA) +\
                            '   B  ' + str(c2w_RB) + '   c  ' + str(c2w_RC)
                robot_str_w2c = 'robot w2c  ' + '   X  ' + str(w2c_RX) + '   Y  ' + str(w2c_RY) + '  Z   ' + str(w2c_RZ) + '   A   ' + str(w2c_RA) +\
                            '   B  ' + str(w2c_RB) + '   c  ' + str(w2c_RC)

                image = cv2.putText(image, robot_str_w2c, (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 4, cv2.LINE_AA)
                image = cv2.putText(image, robot_str_c2w, (40, 160), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 4, cv2.LINE_AA)

                try:
                    Rvec, Tvec = pose_esitmation(frame=image, aruco_dict=aruco_dict, matrix_coefficients=camera_matrix,
                                                 distortion_coefficients=dist_coeff)


                    rot_marker,_ = cv2.Rodrigues(Rvec)

                    angle_marker_w2c = rotation_angles(rot_marker, 'zyx')  # zxy
                    w2c_marA = round(angle_marker_w2c[0], 5)
                    w2c_marB = round(angle_marker_w2c[1], 5 )
                    w2c_marC = round(angle_marker_w2c[2], 5 )

                    w2c_arX =  round(Tvec[0][0][0],5)
                    w2c_arY =  round(Tvec[0][0][1],5)
                    w2c_arZ =  round(Tvec[0][0][2],5)

                    rot_marker_w2c = rotation_matrix(w2c_marA, w2c_marB, w2c_marC, order='zyx')
                    Tm_w2c = np.zeros((4, 4))
                    Tm_w2c[0, 3] = w2c_arX
                    Tm_w2c[1, 3] = w2c_arY
                    Tm_w2c[2, 3] = w2c_arZ
                    Tm_w2c[:3, :3] = rot_marker_w2c
                    Tm_w2c[3, 3] = 1

                    # ##### c2W
                    Tm_c2w = np.linalg.inv(Tm_w2c)

                    rot_marker_c2w = Tm_c2w[:3, :3]
                    c2w_arX = round(Tm_c2w[0, 3],5)
                    c2w_arY = round(Tm_c2w[1, 3],5)
                    c2w_arZ = round(Tm_c2w[2, 3],5)
                    angle_marker = rotation_angles(rot_marker_c2w, 'zyx')  # zxy
                    c2w_marA = round(angle_marker[0], 5)
                    c2w_marB = round(angle_marker[1], 5 )
                    c2w_marC = round(angle_marker[2], 5 )

                    marker_str_w2c = 'marker_w2c' +  '  X  ' + str( w2c_arX)   + '   Y  ' + str( w2c_arY)  + '  Z   ' + str( w2c_arZ) + \
                                 '   A  ' + str( w2c_marA)  + '   B  ' + str( w2c_marB) + '  c   ' + str( w2c_marC)
                    marker_str_c2w = 'marker_c2w' +  '  X  ' + str(c2w_arX)   + '   Y  ' + str(c2w_arY)  + '  Z   ' + str(c2w_arZ) + \
                                 '   A  ' + str(c2w_marA)  + '   B  ' + str(c2w_marB) + '  c   ' + str(c2w_marC)

                    image = cv2.putText(image, marker_str_w2c, (40,40), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255,0,0), 4, cv2.LINE_AA)
                    image = cv2.putText(image,marker_str_c2w, (40,120), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255,0,0), 4, cv2.LINE_AA)

                except:
                    pass


                image = cv2.line(image,  (int(width / 2), int(height / 2)), (int(width / 2), int(height / 2) + 200),  (0, 255, 0), 4)
                image = cv2.line(image,  (int(width / 2), int(height / 2)), (int(width / 2) + 200, int(height / 2)),  (0, 0, 255), 4)
                image = cv2.circle(image, (int(width / 2), int(height / 2)), 5, (255, 0, 0), -1)


                cv2.namedWindow("title", cv2.WINDOW_NORMAL)
                cv2.imshow("title", image)
                k = cv2.waitKey(1) & 0xFF
                # if key pressed, save image in full resolution

                if k == 27 or k == ord('q'):  # break on ESC key or "q"
                    break

        # grabResult.Release()

    finally:
        client.disconnect()
        # Releasing the resource
        camera.StopGrabbing()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()