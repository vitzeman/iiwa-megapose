import json
import os
import time
from datetime import datetime
import copy
import argparse

import cv2
from pypylon import pylon

import numpy as np

import logging


# logging just from this file
logging.getLogger(__name__)

#TODO: Add that the logger shows only this logger
# logging.getLogger().setLevel(logging.INFO)


# logging.basicConfig(
    # format="%(levelname)s %(asctime)s - %(message)s", level=logging.INFO
# )

def parse_arguments():
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--serial_number",
        type=str,
        default="24380112",
        help="Serial number of the camera",
    )
    parser.add_argument(
        "-c",
        "--camera_parameters",
        type=str,
        default=None,
        help="Path to camera parameters",
    )
    parser.add_argument(
        "-l",
        "--save_location",
        type=str,
        default=os.path.join("camera", "images", "demo"),
        help="Path to save images",
    )
    return parser.parse_args()


class BaslerCamera:
    """Class for connectiong basler camera"""

    def __init__(
        self,
        serial_number: str = "24380112",
        camera_parametes: str = None,
        save_location: str = "",
    ) -> None:
        """Initialize basler camera

        Args:
            serial_number (str, optional): Serial number of basler camera (found at the top of the camera S/N). Defaults to "24380112".
            camera_parametes (str, optional): Path to camera parameters. Defaults to None.
            save_location (str, optional): Path to save images. Defaults to "".

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
                f"Camera with serial number {serial_number} not found."
            )

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed

        if save_location != "":
            os.makedirs(save_location, exist_ok=True)

        self.save_location = save_location

        if camera_parametes is not None:
            self.camera_params = self.load_camera_config(
                path_to_config="camera_config.json"
            )
        else:
            self.camera_params = None

        self.camera_ideal_params = None

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

    def process_key(self, key: int) -> bool:
        """Process key pressed by user

        Args:
            key (int): Key pressed by user

        Returns:
            bool: True if program should stop, False otherwise
        """
        if key in [ord("q"), 27]: # quit program maybe add Esc
            return True
        elif key == ord("s"):  # save image
            self.save_current_image()
        elif key == ord("a"):  # automatic camera adjustment
            self.adjust_camera()
        elif key == ord("h"):  # print help
            logging.info("Camera control help:")
            print("\t<q> - Quit")
            print("\t<s> - Save image")
            print("\t<a> - Automatic camera adjustment")
            print("\t<h> - Print help")

        return False

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
        # TODO: Add type checking
        my_config = {}
        for key, val in config.items():
            if key in ["h", "H", "height"]:
                if val < 0:
                    raise ValueError("Height cannot be negative")
                if type(val) not in [int, float]:
                    raise TypeError("Height must be a number")
                if type(val) == float and val.is_integer() is False:
                    logging.warning("Height is float, but not integer. Casting to int.")
                    val = int(val)
                my_config["h"] = int(val)

            elif key in ["w", "W", "width"]:
                if val < 0:
                    raise ValueError("Width cannot be negative")
                if type(val) not in [int, float]:
                    raise TypeError("Width must be a number")
                if type(val) == float and val.is_integer() is False:
                    logging.warning("Width is float, but not integer. Casting to int.")
                    val = int(val)
                my_config["w"] = val

            elif key in ["K", "camera_matrix"]:
                if type(val) not in [list, np.ndarray]:
                    raise TypeError("Camera matrix must be a list")
                my_config["K"] = np.aray(val)
            elif key in ["dist", "distortion_coefficients"]:
                if type(val) not in [list, np.ndarray]:
                    raise TypeError("Distortion coefficients must be a list")
                my_config["dist"] = np.array(val)
            elif key in ["roi", "ROI", "region_of_interest"]:
                if type(val) not in [list, np.ndarray]:
                    raise TypeError("Region of interest must be a list")
                if len(val) != 4:
                    raise ValueError(
                        "Region of interest must be a list of length 4 (x, y, w, h)"
                    )
                my_config["roi"] = np.array(val)
            else:
                logging.warning(f"Unknown key in camera configuration file: {key}")
                my_config[key] = val

        self.camera_params = copy.deepcopy(config)

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

    def save_current_image(self, name: str = ""):
        """Saves current image from camera to file"""
        image = self.get_single_image()
        if image is None:
            return None

        if name == "":
            name = f"{datetime.now().isoformat()}.png"

        path_to_save = os.path.join(self.save_location, name)
        cv2.imwrite(path_to_save, image)
        logging.info("Image saved to {}".format(path_to_save))

    def save_image(self, image: np.ndarray, name: str = "") -> bool:
        """Save image to file

        Args:
            image (np.ndarray): Image to save
            name (str, optional): Name of the image if not given, current time is used.

        Returns:
            bool: True if image was saved, False otherwise
        """
        if type(image) != np.ndarray:
            logging.warning("Image is not a numpy array, returning None")
            return False

        if name == "":
            name = f"{datetime.now().isoformat()}.jpg"

        path_to_save = os.path.join(self.save_location, name)
        cv2.imwrite(path_to_save, image)
        logging.info("Image saved to {}".format(path_to_save))
        return True

    def get_ideal_camera_parameters(self):
        """Get ideal camera parameters"""
        if self.camera_params is None:
            logging.warning("Camera parameters not loaded, returning None")
            return None
        w, h = self.camera_params["w"], self.camera_params["h"]
        K = self.camera_params["K"]
        dist = self.camera_params["dist"]

        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        _, _, new_w, new_h = roi
        new_ideal_camera_params = {
            "K": new_K,
            "dist": np.array([0, 0, 0, 0, 0]),
            "roi": roi,
            "w": new_w,
            "h": new_h,
        }

    def undistort_image(self, image):
        """Undistort image using camera parameters"""
        if self.camera_params is None:
            logging.warning("Camera parameters not loaded, returning None")
            return None

        if self.camera_ideal_params is None and self.camera_params is not None:
            self.camera_ideal_params = self.get_ideal_camera_parameters()

        undistorted_image = cv2.undistort(
            image,
            self.camera_params["K"],
            self.camera_params["dist"],
            None,
            self.camera_ideal_params["K"],
        )
        x, y, w, h = self.camera_ideal_params["roi"]
        undistorted_image = undistorted_image[y : y + h, x : x + w]
        return undistorted_image

    # TODO: test this function
    def get_image_stream(self):
        """Get image stream from camera"""
        if self.camera is None:
            raise pylon.RuntimeException("Camera not connected.")

        while True:
            grabResult = self.camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )

            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                image = image.GetArray()
                grabResult.Release()
                yield image
            else:
                logging.warning("Image not grabbed, returning None")
                grabResult.Release()
                yield None


# ---------------------------------------------------------------------------- #
# Few demos of usage
def manually_capture_images(
    save_location: str = os.path.join("camera", "images", "demo")
) -> None:
    """Manually capture images from camera and saves them to file"""
    camera = BaslerCamera(save_location=save_location)
    camera.connect()
    camera.adjust_camera()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    while True:
        image = camera.get_single_image()

        cv2.imshow("image", image)

        key = cv2.waitKey(100)
        should_break = camera.process_key(key)

        # End the program if the user pressed 'q' or closed the window
        if should_break or cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) < 1:
            logging.info("Exiting program")
            break

    cv2.destroyAllWindows()
    camera.disconnect()

# Does not work try home with different camera
def record_video(
    save_location: str = os.path.join("camera", "videos", "demo"),
    video_name: str = "demo.mp4",
    fps: int = 10,
) -> None:
    """Record video from camera and saves it to file"""
    camera = BaslerCamera(save_location=save_location)
    camera.connect()
    camera.adjust_camera()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    image = camera.get_single_image()
    resolution = image.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(
        os.path.join(save_location, video_name), fourcc, fps, frameSize=(500, 500)
    )
    while True:
        image = camera.get_single_image()

        cv2.imshow("image", image)
        
        key = cv2.waitKey(int(1/fps * 1000))
        should_break = camera.process_key(key)
        out.write(image)
        # End the program if the user pressed 'q' or closed the window
        if should_break or cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) < 1:
            logging.info("Exiting program")
            break
    
    cv2.destroyAllWindows()
    out.release()
    camera.disconnect()


if __name__ == "__main__":
    args = parse_arguments()
    save_location = args.save_location
    os.makedirs(save_location, exist_ok=True)
    manually_capture_images(save_location=save_location)
    