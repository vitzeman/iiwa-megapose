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

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2) * 0.024
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

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
        head, tail = os.path.split(os.path.split(os.getcwd())[0])
        config_dir = os.path.join(head, tail, 'config')
        camera_file = open(os.path.join(config_dir, 'camera.json'))
        camera_dict = json.load(camera_file)
        camera_matrix = np.array(camera_dict['camera_matrix'])
        dist_coeff = np.array(camera_dict['dist_coeff'])


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
                ######

                robot_str_c2w = 'robot c2w  ' + '   X  ' + str(c2w_RX) + '   Y  ' + str(c2w_RY) + '  Z   ' + str(c2w_RZ) + '   A   ' + str(c2w_RA) +\
                            '   B  ' + str(c2w_RB) + '   c  ' + str(c2w_RC)
                robot_str_w2c = 'robot w2c  ' + '   X  ' + str(w2c_RX) + '   Y  ' + str(w2c_RY) + '  Z   ' + str(w2c_RZ) + '   A   ' + str(w2c_RA) +\
                            '   B  ' + str(w2c_RB) + '   c  ' + str(w2c_RC)
                image = cv2.putText(image, robot_str_w2c, (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 4, cv2.LINE_AA)
                image = cv2.putText(image, robot_str_c2w, (40, 160), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 4, cv2.LINE_AA)


                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray, (10, 7), None)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    # Find the rotation and translation vectors.
                    ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeff)

                    # # project 3D points to image plane
                    # imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coeff)

                    image = cv2.drawFrameAxes(image, camera_matrix, dist_coeff, rvecs, tvecs,0.05)


                    w2c_marA = round(math.degrees(rvecs[0]), 5)
                    w2c_marB = round(math.degrees(rvecs[1]), 5 )
                    w2c_marC = round(math.degrees(rvecs[2]), 5 )

                    w2c_arX =  round(tvecs[0][0], 5)
                    w2c_arY =  round(tvecs[1][0], 5)
                    w2c_arZ =  round(tvecs[2][0], 5)

                    rot_marker_w2c = rotation_matrix(w2c_marA, w2c_marB, w2c_marC, order='zyx')
                    Tm_w2c = np.zeros((4, 4))
                    Tm_w2c[0, 3] = w2c_arX
                    Tm_w2c[1, 3] = w2c_arY
                    Tm_w2c[2, 3] = w2c_arZ
                    Tm_w2c[:3, :3] = rot_marker_w2c
                    Tm_w2c[3, 3] = 1

                    Tm_c2w = np.linalg.inv(Tm_w2c)

                    rot_marker_c2w = Tm_c2w[:3, :3]
                    c2w_arX = round(Tm_c2w[0, 3],5)
                    c2w_arY = round(Tm_c2w[1, 3],5)
                    c2w_arZ = round(Tm_c2w[2, 3],5)
                    angle_marker = rotation_angles(rot_marker_c2w, 'zyx')  # zxy
                    c2w_marA = round(angle_marker[0], 5)
                    c2w_marB = round(angle_marker[1], 5 )
                    c2w_marC = round(angle_marker[2], 5 )

                    ########

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