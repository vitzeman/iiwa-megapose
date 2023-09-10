import logging
import time
from opcua.common.node import Node
from opcua import Client
from pypylon import pylon
import cv2
import os
import argparse
import numpy as np
import json
from utils.robot_motion import path, SubHandler

# get instance of the pylon TransportLayerFactory
tlf = pylon.TlFactory.GetInstance()
devices = tlf.EnumerateDevices()
bop_cam = None

serial_number = "24380112"

for d in devices:
    if d.GetSerialNumber() == serial_number:
        bop_cam = d

camera = pylon.InstantCamera(tlf.CreateDevice(bop_cam))
camera.Open()

## Set things to auto for best image possible
camera.GainAuto.SetValue("Once")
camera.ExposureAuto.SetValue("Once")
camera.BalanceWhiteAuto.SetValue("Once")

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed


if __name__ == "__main__":
    client = Client("opc.tcp://localhost:5000/")
    client.connect()
    client.load_type_definitions()
    root = client.get_root_node()
    objects = client.get_objects_node()
    uri = "http://iiwa-control.ciirc.cz"
    idx = client.get_namespace_index(uri)
    rootObj = objects.get_child(["IIWA"])
    input = rootObj.get_child(["RobotGenericInput"])
    output = rootObj.get_child(["RobotGenericOutput"])
    handler = SubHandler()
    subscription = client.create_subscription(100, handler)
    subscription.subscribe_data_change(
        [
            input.get_child(["ActualX"]),
            input.get_child(["ActualY"]),
            input.get_child(["ActualZ"]),
            input.get_child(["ActualRA"]),
            input.get_child(["ActualRB"]),
            input.get_child(["ActualRC"]),
            input.get_child(["OperationStarted"]),
            input.get_child(["OperationFinished"]),
            input.get_child("Failed"),
        ]
    )

    subscription.subscribe_events(rootObj.get_child(["InMotionEvent"]))

    subscription.subscribe_events(rootObj.get_child(["FailEvent"]))

    # Home position from myin.py
    # handler.move_to_position_with_points(input, X=550, Y=0, Z=400, RA=360, RB=0, RC=270)

    # handler.move_to_position_with_points(input, X=30, Y=0, Z=-400, RA=180, RB=0, RC=90)

    # This works for some reason
    handler.move_to_position_with_points(
        input, X=-200.0, Y=-500.0, Z=500.0, RA=90, RB=0, RC=90
    )
    handler.move_to_position_with_points(
        input, X=-200.0, Y=-500.0, Z=300.0, RA=90, RB=0, RC=90
    )
    # transf, RX, RY, RZ, RA, RB, RC = handler.get_current_pos_base(input)
    print("FInished moving")
    client.disconnect()
