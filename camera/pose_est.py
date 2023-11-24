import cv2 
import numpy as np

from basler_camera import BaslerCamera

marker_size = 0.176


camera_dict = {
  "camera_matrix": [
    [
      2284.1930174276913,
      0.0,
      1240.9841718279845
    ],
    [
      0.0,
      2290.6452749906466,
      991.0143784551314
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "distortion_coefficients": [
    [
      -0.05488170091475803,
      0.05404599709705145,
      0.0005248664001609038,
      -0.0011546849712864757,
      0.004864683745407868
    ]
  ],
  "reprojection_error": 0.06449788738455538,
  "height": 2048,
  "width": 2448
}

def pose_esitmation(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedCandidates = detector.detectMarkers(gray)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    
    camera_matrix = np.array(camera_dict['camera_matrix'])
    dist_coeffs = np.array(camera_dict['distortion_coefficients'])

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    i = 0   
    if ids is not None:
      for id in ids:
          corner = corners[i]
          nada, R, t = cv2.solvePnP(marker_points, corner, camera_matrix, dist_coeffs)
          cv2.putText(frame, str(id[0]), (int(corner[0][0][0]), int(corner[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
          i += 1

          cv2.drawFrameAxes(image=frame, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs, rvec=R, tvec=t, length=0.07)

          print(R, t)

    w,h = frame.shape[:2]
    cv2.putText(frame, str(t), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (h//2, w//2), 5, (0, 0, 255), -1)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        return True
    
    return False
    

if __name__ == "__main__":
    
    camera = BaslerCamera()
    camera.connect()
    camera.adjust_camera()
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    while True:
        frame = camera.get_single_image()
        should_quit = pose_esitmation(frame)

        # cv2.imshow("frame", frame)
        if should_quit:
            break