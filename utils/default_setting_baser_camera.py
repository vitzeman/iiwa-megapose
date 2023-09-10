import pypylon.pylon as py

# get instance of the pylon TransportLayerFactory
tlf = py.TlFactory.GetInstance()
devices = tlf.EnumerateDevices()
bop_cam = None

serial_number = '24380112'

for d in devices:
    if d.GetSerialNumber() == serial_number:
        bop_cam = d

camera = py.InstantCamera(tlf.CreateDevice(bop_cam))
camera.Open()

camera.UserSetSelector = "Default"
camera.UserSetLoad.Execute()

