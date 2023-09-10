import json
import os
import time

import cv2
from pypylon import pylon

import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


# TODO LIST:
# 1. Add function to get stream of images from camera

class BaslerCamera:
    """Class for connectiong basler camera"""

    def __init__(
        self, serial_number: str = "24380112", camera_parametes: str = None
    ) -> None:
        """Initialize basler camera

        Args:
            serial_number (str, optional): Serial number of basler camera (found at the top of the camera S/N). Defaults to "24380112".

        Raises:
            pylon.RuntimeException: If no camera is present in the network
            pylon.RuntimeException: If camera with given serial number is not found
        """
        tlf = pylon.TlFactory.GetInstance()
        devices = tlf.EnumerateDevices()

        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")

        self.camera = None
        for device in devices:
            if device.GetSerialNumber() == serial_number:
                self.camera = pylon.InstantCamera(tlf.CreateDevice(device))
                break

        if self.camera is None:
            raise pylon.RuntimeException(
                "Camera with serial number {} not found.".format(serial_number)
            )

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed

        if camera_parametes is not None:
            self.camera_params = self.load_camera_config(
                path_to_config="camera_config.json"
            )
        else:
            self.camera_params = None

    def connect(self):
        """Connect to camera"""
        self.camera.Open()
        # self.camera.PixelFormat = "RGB8"
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        logging.info("Camera connected")

    def disconnect(self):
        """Disconnect camera"""
        self.camera.Close()
        logging.info("Camera disconnected")

    def adjust_camera(self):
        """Adjust camera settings for best image quality"""
        logging.info("Adjusting camera settings")
        self.camera.GainAuto.SetValue("Once")
        self.camera.ExposureAuto.SetValue("Once")
        self.camera.BalanceWhiteAuto.SetValue("Once")
        # NOTE: Look at the wait time so the camera has time to adjust
        time.sleep(1)

    def load_camera_config(self, path_to_config: str):
        """Load camera configuration from file"""
        if not os.path.exists(path_to_config):
            raise FileNotFoundError("Camera configuration file not found")

        with open(path_to_config, "r") as f:
            config = json.load(f)

        self.camera_params = config

    def get_camera_config(self):
        """Get camera configuration"""
        if self.camera_params is None:
            logging.warning("Camera configuration not loaded, returning None")
        return self.camera_params

    def get_single_image(self):
        """Get single image from camera"""
        if self.camera is None:
            raise pylon.RuntimeException("Camera not connected.")

        grabResult = self.camera.RetrieveResult(
            5000, pylon.TimeoutHandling_ThrowException
        )

        if grabResult.GrabSucceeded():
            image = self.converter.Convert(grabResult)
            image = image.GetArray()
            grabResult.Release()
            return image
        else:
            logging.warning("Image not grabbed, returning None")
            grabResult.Release()
            return None
        

    def save_current_image(self, path_to_save: str=""):
        """Saves current image from camera to file"""
        image = self.get_single_image()
        if image is None:
            return None
        if path_to_save == "":
            path_to_save = "image_{}.jpg".format(time.time())
        cv2.imwrite(path_to_save, image)
        logging.info("Image saved to {}".format(path_to_save))

if __name__ == "__main__":
    camera = BaslerCamera()
    camera.connect()
    camera.adjust_camera()
    while True:
        image = camera.get_single_image()
        # image = camera.converter.Convert(image)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("test.jpg", image)
        elif key == ord("a"):
            camera.adjust_camera()
        else:
            continue
    camera.disconnect()
