import cv2

import numpy as np

FILE = "data/testA1.png"

img = cv2.imread(FILE)

dictionary = cv2.aruco.DICT_APRILTAG_36h11

aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)

arucoParams = cv2.aruco.DetectorParameters()
corners, ids, rejected = cv2.aruco.detectMarkers(
    img, aruco_dict, parameters=arucoParams
)

if len(corners) > 0:
    ids = ids.flatten()

    for markerCorner, markerID in zip(corners, ids):
        tl, tr, br, bl = markerCorner.reshape((4, 2))

        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        cv2.line(img, tl, tr, (0, 255, 0), 2)
        cv2.line(img, tr, br, (0, 255, 0), 2)
        cv2.line(img, br, bl, (0, 255, 0), 2)
        cv2.line(img, bl, tl, (0, 255, 0), 2)

        Cx = int((tl[0] + br[0]) / 2.0)
        Cy = int((tl[1] + br[1]) / 2.0)

        cv2.putText(
            img, str(markerID), (Cx, Cy), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 2
        )

    cv2.imwrite("data/testA1_out.png", img)
