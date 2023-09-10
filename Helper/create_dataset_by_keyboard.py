from pypylon import pylon
import cv2
import os

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

## Set things to auto for best image possible
camera.GainAuto.SetValue("Once")
camera.ExposureAuto.SetValue("Once")
camera.BalanceWhiteAuto.SetValue("Once")

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed

head, tail = os.path.split(os.path.split(os.getcwd())[0])
config_dir = os.path.join(head, tail, 'config')
img_dir = os.path.join(head, tail, 'dataset', 'chessboard_calibration', 'images')

if not os.path.isdir(os.path.join(head, tail, 'dataset', 'chessboard_calibration')):
    os.mkdir(os.path.join(head, tail, 'dataset', 'chessboard_calibration'))
    os.mkdir(img_dir)


def main():
    i = 0
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            image = image.GetArray()
            height, width, channel = image.shape

            cv2.namedWindow("title", cv2.WINDOW_NORMAL)
            cv2.imshow("title", image)
            k = cv2.waitKey(1) & 0xFF
            # if key pressed, save image in full resolution

            if k == 27 or k == ord('q'):  # break on ESC key or "q"
                break

            elif k == ord('s'):  # wait for 's' key to save and exit
                cv2.imwrite(os.path.join(img_dir, str(i) + '.jpg'), image)
                i = i+1



if __name__ == '__main__':
    main()