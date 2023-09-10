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

np.set_printoptions(suppress = True)

def main():

    args = parse_args()
    head, tail = os.path.split(os.path.split(os.getcwd())[0])
    dataset_dir =  os.path.join(head, tail, 'dataset')
    img_dir = os.path.join(dataset_dir, args.dataset_dir, 'images')
    pose_dir = os.path.join(dataset_dir,args.dataset_dir, 'pose')
    pose_file_path = os.path.join(pose_dir, 'pose.json')
    config_dir = os.path.join(head, tail, 'config')
    texron_dataset = os.path.join(head, tail,'texron')

    if not os.path.exists(texron_dataset):
        os.mkdir(texron_dataset)

    undistort_img_dir = os.path.join(texron_dataset,os.path.split(args.dataset_dir)[1])
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

        # P = np.eye(4)
        # P[:3, :] = newcameramtx @ W2C[:3, ]
        # P = P[:3, ]

        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        print(os.path.join(undistort_img_dir, key))
        cv.imwrite(os.path.join(undistort_img_dir, key), dst)

        translation = str(W2C[:3,3:].T).strip('[').strip(']')
        rotation1 = str(W2C[0,:3]).strip('[').strip(']')
        rotation2 = str(W2C[0,:3]).strip('[').strip(']')
        rotation3 = str(W2C[0,:3]).strip('[').strip(']')
        focal = str(newcameramtx[0,0]).strip('[').strip(']')
        paspect = str(w/h).strip('[').strip(']')
        ppx = str(newcameramtx[0,2]).strip('[').strip(']')
        ppy = str(newcameramtx[1,2]).strip('[').strip(']')

        line1 = translation[1:] + rotation1 + rotation2 + rotation3
        line2 = focal + ' ' + str(0) + ' ' + str(0) + ' ' + paspect + ' ' + ppx + ' ' + ppy
        print(line1)
        print(line2)

        file_path = os.path.join(undistort_img_dir, img_name + '.cam')
        with open(file_path, 'w') as f:
            f.write(line1)
            f.write('\n' + line2)



if __name__ == '__main__':
    main()