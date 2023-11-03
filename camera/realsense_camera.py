import argparse 
import os
import json 

import cv2
import numpy as np
import pyrealsense2 as rs

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", type=str, help="Path 2 save the recording", default="data")
    args = parser.parse_args()
    return args


class RealSenseCamera:
    """Class for realsesnse camera"""

    def __init__(self) -> None:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise Exception("No device connected, please connect a RealSense device")
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable streams with the same resolution
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Align depth frame to color frame
        self.align = rs.align(rs.stream.color)

        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
        # Used intrinsics 
        self.color_stream = self.profile.get_stream(rs.stream.color)
        self.color_intrinsics = self.color_stream.as_video_stream_profile().get_intrinsics()
        print("Color intrinsics: ", self.color_intrinsics)

    def end_stream(self):
        self.pipeline.stop()

    def get_intrinsics(self):
        return self.color_intrinsics

    def get_aligned_frames(self):
        framset = self.pipeline.wait_for_frames()
        color_frame = framset.get_color_frame()
        depth_frame = framset.get_depth_frame()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(framset)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        return color_frame, aligned_depth_frame
    
    def make_recoring(self, save_path:str):
        save_folder = save_path
        save_color_path = os.path.join(save_folder, "color")
        save_depth_path = os.path.join(save_folder, "depth")
        os.makedirs(save_color_path, exist_ok=True)
        os.makedirs(save_depth_path, exist_ok=True)
        intrinsics = self.get_intrinsics()
        print("Intrinsics: ", type(intrinsics))

        intrinsics_d = {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "height": intrinsics.height,
            "width": intrinsics.width,
            "model": str(intrinsics.model),
            "coeffs": intrinsics.coeffs,
        }
        with open(os.path.join(save_folder, "intrinsics.json"), "w") as f:
            json.dump(intrinsics_d, f, indent=2)
        i = 0
        while True:
            name = str(i).zfill(6)
            color_frame, aligned_depth_frame = self.get_aligned_frames()
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            cv2.imshow("color", color_image)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            cv2.imwrite(os.path.join(save_color_path, name + ".png"), color_image)
            cv2.imwrite(os.path.join(save_depth_path, name + ".png"), depth_image)
            i += 1
            
        
if __name__ == "__main__":
    args = parse_args()
    save_path = args.save_path
    cam = RealSenseCamera()
    cam.make_recoring(save_path)
    # while True:
    #     color_frame, aligned_depth_frame = cam.get_aligned_frames()
    #     color_image = np.asanyarray(color_frame.get_data())
    #     depth_image = np.asanyarray(aligned_depth_frame.get_data())
    #     cv2.imshow("color", color_image)
    #     cv2.imshow("depth", depth_image)
    #     # print("Color image shape: ", color_image.shape, "Depth image shape: ", depth_image.shape)

    #     # print(color_image.dtype, depth_image.dtype)
    #     key = cv2.waitKey(1)
    #     if key == ord('q'):
    #         break
    #     elif key == ord('s'):
    #         cv2.imwrite("color.png", color_image)
    #         cv2.imwrite("depth.png", depth_image)