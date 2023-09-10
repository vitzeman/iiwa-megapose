import logging
import time
from opcua.common.node import Node
from opcua import Client
from pypylon import pylon
import cv2
import os
import argparse
import numpy as np
import json
from utils.robot_motion import path, SubHandler

# get instance of the pylon TransportLayerFactory
tlf = pylon.TlFactory.GetInstance()
devices = tlf.EnumerateDevices()
bop_cam = None

serial_number = '24380112'

for d in devices:
    if d.GetSerialNumber() == serial_number:
        bop_cam = d

camera = pylon.InstantCamera(tlf.CreateDevice(bop_cam))
camera.Open()

## Set things to auto for best image possible
camera.GainAuto.SetValue("Once")
camera.ExposureAuto.SetValue("Once")
camera.BalanceWhiteAuto.SetValue("Once")

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, required=True, help="specify the name of the object")
args = parser.parse_args()

home = {'A': -160, 'B': 50, 'C': -130, 'z': 850, 'y': -750, 'x': -350}

# orientation = {
#               -90: {"RB": {"S": -1, "P": 3, }, "RC": {"S": -1, "P": 2}},
#                0: {"RB": {"S": 1, "P": 2 }, "RC": {"S": -1, "P": 3}},
#                180: {"RB": {"S": -1, "P": 2}, "RC": {"S": 1, "P": 3}},
#                90: {"RB": {"S": 1, "P": 3}, "RC": {"S": 1, "P": 2}},
#      }

#for calibration
orientation = {
              # -90: {"RB": {"S": -1, "P": 3, }, "RC": {"S": -1, "P": 2}},
              180: {"RB": {"S": -1, "P": 2}, "RC": {"S": 1, "P": 3}},
              #  0: {"RB": {"S": 1, "P": 2 }, "RC": {"S": -1, "P": 3}},
               # 90: {"RB": {"S": 1, "P": 3}, "RC": {"S": 1, "P": 2}},


     }

last_pos = np.array((0, 0, 0))

def create_dataset_dir():
    if not os.path.isdir('dataset'):
        os.mkdir('dataset')
    dataset_dir = os.path.join('dataset', args.name)
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    img_dir = os.path.join(dataset_dir, 'images')
    pose_dir = os.path.join(dataset_dir ,'pose')
    config_dir = os.path.join(dataset_dir ,'pose')
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    if not os.path.isdir(pose_dir):
        os.mkdir(pose_dir)
    if not os.path.isdir(config_dir):
        os.mkdir(config_dir)

    return img_dir, pose_dir, config_dir

def load_camera():
    root = os.getcwd()
    config_dir = os.path.join(root, 'config')
    camera_file = open(os.path.join(config_dir, 'camera.json'))
    camera = json.load(camera_file)
    return camera

def check_point_before(point_check, point_list):
    dist_list = []
    for point in point_list:
        dist = np.linalg.norm(point - point_check)
        dist_list.append(dist)

    if min(dist_list) < 1:
        print("already been to that point")
        return False
    else:
        print("Havent been to point yet")
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    client = Client("opc.tcp://localhost:5000/")
    img_dir, pose_dir, config_dir = create_dataset_dir()
    camera_config = load_camera()
    pose_dict = {}
    raw_translation_rotation = {}
    camera_dict = {}

    camera_matrix = np.array(camera_config['camera_matrix'])
    dist_coeff = np.array(camera_config['dist_coeff'])

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
        handler = SubHandler()
        subscription = client.create_subscription(100, handler)
        subscription.subscribe_data_change([input.get_child(["ActualX"]),
                                            input.get_child(["ActualY"]),
                                            input.get_child(["ActualZ"]),
                                            input.get_child(["ActualRA"]),
                                            input.get_child(["ActualRB"]),
                                            input.get_child(["ActualRC"]),
                                            input.get_child(["OperationStarted"]),
                                            input.get_child(["OperationFinished"]),
                                            input.get_child("Failed")])

        subscription.subscribe_events(rootObj.get_child(['InMotionEvent']))

        subscription.subscribe_events(rootObj.get_child(['FailEvent']))

        frame = 0
        all_points = []

        # for dataset
        distance = 300
        z_offset = 0
        motion = path(center_x= 550, center_y= 0, number_points= 20, distance = distance, min_height= 270, height_change= 30, z_offset = z_offset, max_height= distance)

        # for calibration
        # motion = path(center_x= -200, center_y= -500, number_points= 16, distance = 300, min_height=0, height_change= 30, z_offset = z_offset, max_height= 300)

        points1 = motion.get_circlular_points()
        all_points.extend(points1)
        print("total number of points", len(all_points))
        completed_points = [np.array((0,0))]


        for orient in orientation:
            point_num = 1
            for point in all_points:
                point_try = np.array((int(point[0]),int(point[1])))
                if check_point_before(point_try, completed_points):
                    print("point", point_num," of ",len(all_points),  " in orientation", orient)
                    handler.move_to_position_with_points(input, X=point[0], Y=point[1], Z=point[4], RA = orient,
                                                RB= orientation[orient]["RB"]["S"]* point[orientation[orient]["RB"]["P"]],
                                                RC= 180 + orientation[orient]["RC"]["S"]* point[orientation[orient]["RC"]["P"]] )

                    # # # when camera is vertical
                    # # Front
                    # operation = handler.move_to_position_with_points(input, X=point[0], Y=point[1], Z=point[4], RA = -90, RB= -point[3],
                    #                              RC= 180 - point[2])

                    # # Back
                    # operation = handler.move_to_position_with_points(input, X=point[0], Y=point[1], Z=point[4], RA = 90, RB= point[3],
                    #                              RC= 180 + point[2])

                    # # left
                    # operation = handler.move_to_position_with_points(input, X=point[0], Y=point[1], Z=point[4], RA = 0, RB= point[2],
                    #                              RC= 180 - point[3])

                    # # right
                    # operation = handler.move_to_position_with_points(input, X=point[0], Y=point[1], Z=point[4], RA = 180, RB= -point[2],
                    #                              RC= 180 + point[3] )

                    operation = handler.check_point_fail_pass(input)
                    transf, RX,RY, RZ, RA, RB, RC = handler.get_current_pos_base(input)
                    raw_translation_rotation[frame] = {}
                    raw_translation_rotation[frame]['RX'] = RX
                    raw_translation_rotation[frame]['RY'] = RY
                    raw_translation_rotation[frame]['RZ'] = RZ
                    raw_translation_rotation[frame]['RA'] = RA
                    raw_translation_rotation[frame]['RB'] = RB
                    raw_translation_rotation[frame]['RC'] = RC
                    current_pos = np.array((RX, RY, RZ))
                    if operation:
                        print('successful frame', frame )
                        fname = str(frame) + '.jpg'
                        pose_dict[fname] = {}
                        pose_dict[fname]['K'] = camera_matrix.tolist()
                        pose_dict[fname]['img_size'] = camera_config['img_size']
                        pose_dict[fname]['W2C'] = transf.tolist()

                        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                        if camera.IsGrabbing():
                            if grabResult.GrabSucceeded():
                                image = converter.Convert(grabResult)
                                img = image.GetArray()
                                cv2.imwrite(os.path.join(img_dir, str(frame) + ".jpg"), img)
                                frame += 1

                        point_done = np.array((int(point[0]), int(point[1])))
                        completed_points.append(point_done)

                        if orient == 180:
                            # move to home to avoid collision
                            handler.move_to_position_with_points(input, X= 550, Y= 0, Z= 400,RA= 180,
                                                                 RB=0,
                                                                 RC=90)


                    else:
                        if np.linalg.norm(current_pos - last_pos) > 2:
                            print("moving to home position")

                            if orient == 90:
                                # move to home to avoid collision
                                handler.move_to_position_with_points(input, X=550, Y=0, Z= 400, RA= 90,
                                                                     RB=0,
                                                                     RC=180 )
                                # operation = handler.check_point_fail_pass(input)
                                # transf, RX, RY, RZ, RA, RB, RC = handler.get_current_pos_base(input)
                                last_pos = np.array((RX, RY, RZ))

                            if orient == -90:
                                # move to home to avoid collision
                                handler.move_to_position_with_points(input, X=550, Y=0, Z= 400, RA= -90,
                                                                     RB=0,
                                                                     RC=180)
                                # operation = handler.check_point_fail_pass(input)
                                # transf, RX, RY, RZ, RA, RB, RC = handler.get_current_pos_base(input)
                                last_pos = np.array((RX, RY, RZ))

                            if orient == 180:
                                # move to home to avoid collision
                                handler.move_to_position_with_points(input, X= 550, Y=0, Z= 400,RA= 180,
                                                                     RB=0,
                                                                     RC=90)
                                # operation = handler.check_point_fail_pass(input)
                                # transf, RX, RY, RZ, RA, RB, RC = handler.get_current_pos_base(input)
                                last_pos = np.array((RX, RY, RZ))

                            if orient == 0:
                                handler.move_to_position_with_points(input, X=550, Y=0, Z=400, RA=orient,
                                                                     RB=0,
                                                                     RC=90)
                                last_pos = np.array((RX, RY, RZ))

                        else:
                            print("Already at home position")

                point_num += 1

        with open(os.path.join(pose_dir,"pose.json"), "w") as outfile:
            json.dump(pose_dict, outfile, indent=2)

        with open(os.path.join(pose_dir,"raw.json"), "w") as outfile1:
            json.dump(raw_translation_rotation, outfile1, indent=2)

        with open(os.path.join(config_dir,"camera.json"), "w") as outfile2:
            json.dump(camera_config, outfile2, indent=2)

        print(len(completed_points)-1)

    finally:
        client.disconnect()