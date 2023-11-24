import os
import json
import time 

import numpy as np
import cv2 

from KMR_IIWA.IIWA_robot import IIWA
from KMR_IIWA.IIWA_tools import IIWA_tools
from camera.basler_camera import BaslerCamera


if __name__ == "__main__":
    robot1_ip = "172.31.1.10"
    iiwa = IIWA(robot1_ip)
   
    pos = iiwa.getCartisianPosition(tool=None)
    print(pos)

    camera = BaslerCamera(save_location=os.path.join("camera", "new_capture_ext"))
    camera.connect()
    camera.adjust_camera()
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    while True:
        image = camera.get_single_image()
        
        cv2.imshow("frame", image)

        name = f"{time.strftime('%Y%m%d-%H%M%S')}"
        name_png = f"{name}.png"
        name_json = f"{name}.json"
        key = cv2.waitKey(1)

        should_break = camera.process_key(key, name=name_png)
        if key ==ord("s"):
            pos = iiwa.getCartisianPosition()
            with open(os.path.join("camera", "new_capture_ext", name_json), "w") as f:
                json.dump(pos, f, indent=2)

        if should_break:
            break
