# 1st party
import argparse
import os
import time
import json
from typing import Tuple, Union
import copy

# 3rd party
from mlsocket import MLSocket
import cv2
import numpy as np

# Mine
from cluster.Client import get_megapose_estimation
from camera.basler_camera import BaslerCamera

LABELS = {
    1: "d01_controller",
    2: "d02_servo",
    3: "d03_main",
    4: "d04_motor",
    5: "d05_axle_front",
    6: "d06_battery",
    7: "d07_axle_rear",
    8: "d08_chassis",
}
LABELS_NUMS_KEY = [x + 48 for x in LABELS.keys()] # ASCII code for numbers from 1 to 8


class Frame_processing(BaslerCamera):

    def __init__(self, serial_number: str = "24380112", camera_parametes: str = None, save_location: str = "") -> None:
        super().__init__(serial_number, camera_parametes, save_location)
        
        self.frame = None
        self.bbox = None
        self.idx = None

        self.window_name = "Proccess frame"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.extract_coordinates)

    def reset(self):
        self.frame = None
        self.bbox = None 
        self.idx  = None

    def extract_coordinates(self, event, x, y, flags, parameters):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Left click")
            self.bbox = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            print("Left release")
            self.bbox.extend([x, y])
            self.bbox = np.array(self.bbox)
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("Right click")
            self.bbox = None

    def proccess_frame(self) ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = self.get_single_image()
        frame_vis = copy.deepcopy(frame)
        key = cv2.waitKey(1) & 0xFF
        should_quit = False
        if key == ord("q"):
            should_quit = True
        elif key == ord("h"): #help
            print("q - quit NOT IMPLEMENTED")
            print("h - help")
            print("To select object press number from 1 to 8")
            for key, value in LABELS.items():
                print(f"  {value} - {key}")
            print("To set bbox click on the image and drag the mouse - NOT IMPLEMENTED")
            print("To confirm press Enter")
            print("To reset press r")
        elif key in LABELS_NUMS_KEY:
            self.idx = np.array([key - 48]) # Convert ASCII code to number
            print(f"Selected object: {LABELS[self.idx[0]]}")
        elif key == ord("r"):
            self.reset()
        elif key == 13: # enter
            self.frame = frame

        if self.idx is None:
            cv2.putText(frame_vis, "Selected object: -", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_vis, f"Selected object: {LABELS[self.idx[0]]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        if self.bbox is not None and len(self.bbox) == 4:
            cv2.rectangle(frame_vis, (self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), (0, 255, 0), 2)

        cv2.imshow(self.window_name, frame_vis)
        return should_quit, self.frame, self.bbox, self.idx


if __name__ == "__main__":
    # Camera init
    detector = Frame_processing()
    detector.connect()
    detector.adjust_camera()

    # TODO: Add parser for the host and port
    # Server comunication init
    ml_socket = MLSocket()
    host = "10.35.129.250"
    port = 65432
    ml_socket.connect((host, port))
    print(f"Connection to {host}:{port} established")

    # Main loop
    while True:
        should_quit, frame, bbox, idx = detector.proccess_frame()
        if should_quit: # Should quit after q is pressed in the window
            break

        if frame is None or bbox is None or idx is None:
            continue

        if type(frame) != np.ndarray or type(bbox) != np.ndarray or type(idx) != np.ndarray:
            print("Wrong type")
            continue

        # Get the pose from megapose running on the cluster 
        pose = get_megapose_estimation(ml_socket, frame, bbox, idx)

        # Plan the movement of the robot

    # Deactivate everything else
    ml_socket.close()
    detector.disconnect()
    cv2.destroyAllWindows()
