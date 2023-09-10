from opcua import Client
import numpy as np
import cv2
from pypylon import pylon


camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

## Set things to auto for best image possible
camera.GainAuto.SetValue("Once")
camera.ExposureAuto.SetValue("Once")
camera.BalanceWhiteAuto.SetValue("Once")

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed

points_3d = np.array([[0.0,1.2,0.0],
                      [1.2,1.2,0.0],
                      [1.2,0.0,0.0],
                      [0.0,0.0,0.0]])

points_3d[:,0:2] -= 0.6

# camera_matrix = np.array([[2.30090784e+03, 0.00000000e+00, 1224],
#                           [0.00000000e+00, 2.29285500e+03, 1024],
#                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeff = np.array([0, 0, 0, 0, 0])
focal_length_mm = 2.3
width_pixels = 2448
height_pixels = 2048
width_mm = 176.0
pixels_per_mm = width_pixels/width_mm
focal_length_pixels = pixels_per_mm * focal_length_mm


camera_matrix = np.array([[focal_length_pixels, 0.0 , width_pixels/2],
              [0.0   , focal_length_pixels, height_pixels/2],
              [0.0   , 0.0   ,    1.0]])

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()

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
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.174, matrix_coefficients,
                                                                           distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            # cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return rvec, tvec

def pose_estimation2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Change grayscale
    (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
    for i in range(len(corners)):
        tag_id = int(ids[i])
        points_2d = corners[i].squeeze()

        ret, rvec, tvec = cv2.solvePnP(objectPoints=points_3d, imagePoints=points_2d, cameraMatrix= camera_matrix, distCoeffs=dist_coeff,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)

    return rvec, tvec

def main():
    # client = Client("opc.tcp://localhost:5000/")
    try:

        # client.connect()
        # client.load_type_definitions()
        # root = client.get_root_node()
        # objects = client.get_objects_node()
        # uri = "http://iiwa-control.ciirc.cz"
        # idx = client.get_namespace_index(uri)
        # rootObj = objects.get_child(["IIWA"])
        # input = rootObj.get_child(["RobotGenericInput"])
        # output = rootObj.get_child(["RobotGenericOutput"])
        # print("connection started")


        while camera.IsGrabbing():
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                image = image.GetArray()

                Rvec, Tvec = pose_esitmation(frame=image, aruco_dict=arucoDict, matrix_coefficients=camera_matrix,
                                             distortion_coefficients=dist_coeff)

                Rvecpnp, Tvecpnp = pose_estimation2(image)
                # print(Rvecpnp, Tvecpnp)

                arX = round(Tvec[0][0][0], 5)
                arY = round(Tvec[0][0][1], 5)
                arZ = round(Tvec[0][0][2], 5)
                marker_str = '   X  ' + str(arX) + '   Y  ' + str(arY) + '  Z   ' + str(arZ)
                image = cv2.putText(image, 'marker' + marker_str, (40, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 4, cv2.LINE_AA)

                RX = Tvecpnp[0]
                RY = Tvecpnp[1]
                RZ = Tvecpnp[2]

                robot_str = '   X  ' + str(RX) + '   Y  ' + str(RY) + '  Z   ' + str(RZ)


                image = cv2.putText(image, 'pnp  ' + robot_str, (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 4, cv2.LINE_AA)

            cv2.namedWindow("title", cv2.WINDOW_NORMAL)
            cv2.imshow("title", image)
            k = cv2.waitKey(1) & 0xFF
            # if key pressed, save image in full resolution

            if k == 27 or k == ord('q'):  # break on ESC key or "q"
                break


    finally:

        # client.disconnect()
        # Releasing the resource
        camera.StopGrabbing()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

