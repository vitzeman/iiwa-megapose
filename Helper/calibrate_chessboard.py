import numpy as np
import cv2 as cv
import glob
import os
import json
import math

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

np.savez(os.path.join(config_dir, 'B.npz'), mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)



# for fname in images:
#     img = cv.imread(fname)
#     h,  w = img.shape[:2]
#     newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
#
#     # undistort
#     dst = cv.undistort(img, mtx, dist, None, newcameramtx)
#     # crop the image
#     x, y, w, h = roi
#     dst = dst[y:y+h, x:x+w]
#
#     dst = cv.resize(dst, (960, 540))
#     # Using cv2.imshow() method
#     # Displaying the image
#     cv.imshow("window_name", dst)
#
#     # waits for user to press any key
#     # (this is necessary to avoid Python kernel form crashing)
#     cv.waitKey(0)
#
#     # closing all open windows
#     cv.destroyAllWindows()
