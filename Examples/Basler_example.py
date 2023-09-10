from pypylon import pylon
import cv2

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed

camera.GainAuto.SetValue("Once")
camera.ExposureAuto.SetValue("Once")
camera.BalanceWhiteAuto.SetValue("Once")

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():

        image = converter.Convert(grabResult)
        img = image.GetArray()
        cv2.imshow("window", img)

