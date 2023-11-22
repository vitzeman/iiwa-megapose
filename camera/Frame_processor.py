import os
import copy
import json
from typing import Tuple

import cv2
import numpy as np

from camera.basler_camera import BaslerCamera

# TODO: Maybe try to replace it with some constant/config file
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


ASCII_NUM_SHIFT = 48
LABELS_NUMS_KEY = [x + ASCII_NUM_SHIFT for x in LABELS.keys()]


class FrameProcessor(BaslerCamera):
    """Class for processing frames(manual input of bbox and object type) from camera.
    Also does the visualization of the process.
    Inherits from BaslerCamera class.
    """

    def __init__(
        self,
        serial_number: str = "24380112",
        camera_parametes: str = os.path.join("camera_parameters.json"),
        save_location: str = "",
    ) -> None:
        super().__init__(serial_number, camera_parametes, save_location)
        self.camera_ideal_params = self.get_ideal_camera_parameters()

        self.frame = None
        self.bbox = None
        self.idx = None

        self.window_name = "Proccess frame"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.extract_coordinates)

    def reset(self):
        self.frame = None
        self.bbox = None
        self.idx = None

    def extract_coordinates(self, event, x, y, flags, parameters):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print("Left click")
            self.bbox = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            # print("Left release")
            self.bbox.extend([x, y])
            self.bbox = np.array(self.bbox)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # print("Right click")
            self.bbox = None

    def proccess_frame(self) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        """Method to process frame from camera. Gets frame, undistorts it,
        shows it to user and waits for user to select object and bbox.
        After that returns frame, bbox and object type.

        Returns:
            Tuple[bool, np.ndarray, np.ndarray, np.ndarray]: user input to quit, frame(), bounding box(4,) and object type(1,)
        """
        frame = self.get_single_image()
        frame = self.undistort_image(frame)
        frame_vis = copy.deepcopy(frame)
        # print(frame.shape)  # TODO: remove hera and update the docstring 2035x2441x3
        key = cv2.waitKey(1) & 0xFF
        should_quit = False
        if key == ord("q"):
            should_quit = True
        elif key == ord("h"):  # help
            print("#--HELP------------------------------------------------------#")
            print("q -> quit")
            print("h -> help")
            print("r -> reset")
            print("1-8 -> select object:")
            for key, value in LABELS.items():
                print(f"    {key} - {value}")
            print("To set bbox click on the image and drag the mouse")
            print("To confirm and run the inference press Enter")
            print("#------------------------------------------------------------#")
        elif key in LABELS_NUMS_KEY:
            self.idx = np.array([key - ASCII_NUM_SHIFT])  # Convert ASCII code to number
            print(f"Selected object: {LABELS[self.idx[0]]}")
        elif key == ord("r"):
            self.reset()
        elif key == 13:  # enter
            if self.idx is None or self.bbox is None:
                print("Please select object and bbox")
            else:
                self.frame = frame
                # TODO: does not work for some reason
                frame_vis = cv2.addWeighted(
                    frame_vis, 0.5, np.zeros_like(frame_vis), 0.5, 0
                )
                frame_vis = cv2.putText(
                    frame_vis,
                    f"Running_inference on {LABELS[self.idx[0]]}",
                    (400, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        if self.idx is None:
            cv2.putText(
                frame_vis,
                "Selected object: -",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame_vis,
                f"Selected object: {LABELS[self.idx[0]]}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if self.bbox is not None and len(self.bbox) == 4:
            cv2.rectangle(
                frame_vis,
                (self.bbox[0], self.bbox[1]),
                (self.bbox[2], self.bbox[3]),
                (0, 255, 0),
                5,
            )

        cv2.imshow(self.window_name, frame_vis)

        return should_quit, self.frame, self.bbox, self.idx
