import numpy as np
import cv2 as cv
import glob
import os
import json
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format poses.npy")
    parser.add_argument("-i", "--dataset_dir", required=True, help="input directory ")
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    head, tail = os.path.split(os.path.split(os.getcwd())[0])
    dataset_dir =  os.path.join(head, tail, 'dataset')
    img_dir = os.path.join(dataset_dir, args.dataset_dir, 'images')
    pose_dir = os.path.join(dataset_dir,args.dataset_dir, 'pose')
    pose_file_path = os.path.join(pose_dir, 'pose.json')
    config_dir = os.path.join(head, tail, 'config')
    capture_reality_dataset = os.path.join(head, tail,'capture_reality')

    if not os.path.exists(capture_reality_dataset):
        os.mkdir(capture_reality_dataset)

    undistort_img_dir = os.path.join(capture_reality_dataset,os.path.split(args.dataset_dir)[1])
    if not os.path.exists(undistort_img_dir):
        os.mkdir(undistort_img_dir)

    file = open(pose_file_path)
    data = json.load(file)

    # Load previously saved data
    with np.load(os.path.join(config_dir,'B.npz')) as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    for key in data:
        img_path = os.path.join(img_dir, key)
        img_name = str(key).replace('.jpg', '')

        img = cv.imread(img_path)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        W2C = np.array(data[key]["W2C"])
        P = np.eye(4)
        P[:3, :] = newcameramtx @ W2C[:3, ]
        P = P[:3, ]

        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        print(os.path.join(undistort_img_dir, key))

        cv.imwrite(os.path.join(undistort_img_dir, key), dst)
        file_path = os.path.join(undistort_img_dir, img_name +'_P' + '.txt')
        np.savetxt(file_path, P, fmt='%.2f')

        # dst = cv.resize(dst, (960, 540))
        # # Using cv2.imshow() method
        # # Displaying the image
        # cv.imshow("window_name", dst)
        #
        # # waits for user to press any key
        # # (this is necessary to avoid Python kernel form crashing)
        # cv.waitKey(0)
        #
        # # closing all open windows
        # cv.destroyAllWindows()


    out_dict = {}
    out_dict['camera_matrix'] = newcameramtx.tolist()
    with open(os.path.join(pose_dir, "undistortedcamera.json"), "w") as outfile:
        json.dump(out_dict, outfile, indent= 2)

if __name__ == '__main__':
    main()