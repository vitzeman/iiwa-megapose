
import numpy as np
import cv2
from pypylon import pylon


camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

## Set things to auto for best image possible
camera.GainAuto.SetValue("Once")
camera.ExposureAuto.SetValue("Once")
camera.BalanceWhiteAuto.SetValue("Once")

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
from pupil_apriltags import Detector

at_detector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)

def main():
   while camera.IsGrabbing():
      grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
      if grabResult.GrabSucceeded():
         image = converter.Convert(grabResult)
         image = image.GetArray()
         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         det = at_detector.detect(gray)
         print(det)
         cv2.namedWindow("title", cv2.WINDOW_NORMAL)
         cv2.imshow("title", image)
         k = cv2.waitKey(1) & 0xFF
         # if key pressed, save image in full resolution

         if k == 27 or k == ord('q'):  # break on ESC key or "q"
            break

if __name__ == '__main__':
    main()