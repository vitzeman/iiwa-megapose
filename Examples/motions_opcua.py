import logging
import time
from opcua import Client
import math
import os
import numpy as np


class SubHandler(object):

    def datachange_notification(self, node, val, data):
        # check for change in OperationFinished
        print("Python: New data change event", node, val)

    def event_notification(self, event):
        print("Python: New event", event)

def get_current_pos_base():
        RX = input.get_child(["ActualX"]).get_value()
        RY = input.get_child(["ActualY"]).get_value()
        RZ = input.get_child(["ActualZ"]).get_value()
        RA = input.get_child(["ActualRA"]).get_value()
        RB = input.get_child(["ActualRB"]).get_value()
        RC = input.get_child(["ActualRC"]).get_value()
        print("RX ", round(float(RX),3), "RY ", round(float(RY),3), "RZ ", round(float(RZ),3), "RA ", round(float(RA),3), "RB ", round(float(RB),3), "RC ", round(float(RC),3) )

        matrix = t_matrix()
        matrix.__int__(RX, RY, RZ, RA, RB, RC)
        transformation = matrix.find_transform()

        return transformation


def move_to_position(coordinate):
    if coordinate == "base":
        input.get_child(["ProgramNumber"]).set_value(1)
    if coordinate == "world":
        input.get_child(["ProgramNumber"]).set_value(2)

    # world example
    input.get_child(["DataReady"]).set_value(False)
    input.get_child(["TargetX"]).set_value(420)
    input.get_child(["TargetY"]).set_value(-600)
    input.get_child(["TargetZ"]).set_value(300)
    input.get_child(["TargetRA"]).set_value(math.radians(-90))
    input.get_child(["TargetRB"]).set_value(math.radians(0))
    input.get_child(["TargetRC"]).set_value(math.radians(179))

    # # base example
    # input.get_child(["DataReady"]).set_value(False)
    # input.get_child(["TargetX"]).set_value(0)
    # input.get_child(["TargetY"]).set_value(0)
    # input.get_child(["TargetZ"]).set_value(340)
    # input.get_child(["TargetRA"]).set_value(math.radians(0))
    # input.get_child(["TargetRB"]).set_value(math.radians(0))
    # input.get_child(["TargetRC"]).set_value(math.radians(0))

    # trFrame = input.get_child(["TargetFrame"]).set_value("/Home")
    input.get_child(["DataReady"]).set_value(True)
    time.sleep(4.0)

def move_to_position_with_points(coordinate, X, Y, Z, RA, RB, RC):
    if coordinate == "base":
        input.get_child(["ProgramNumber"]).set_value(1)
    if coordinate == "world":
        input.get_child(["ProgramNumber"]).set_value(2)

    # world example
    input.get_child(["DataReady"]).set_value(False)
    input.get_child(["TargetX"]).set_value(X)
    input.get_child(["TargetY"]).set_value(Y)
    input.get_child(["TargetZ"]).set_value(Z)
    input.get_child(["TargetRA"]).set_value(math.radians(RA))
    input.get_child(["TargetRB"]).set_value(math.radians(RB))
    input.get_child(["TargetRC"]).set_value(math.radians(RC))

    input.get_child(["DataReady"]).set_value(True)
    time.sleep(6.0)
    get_current_pos_base()


class t_matrix:

    def __int__(self, x, y, z, ra, rb, rc):
        self.rotx = np.zeros((3,3))
        self.roty = np.zeros((3,3))
        self.royz = np.zeros((3,3))
        self.T = np.zeros((4,4))
        self.angle_x = ra
        self.angle_y = rb
        self.angle_z = rc
        self.X = x
        self.Y = y
        self.Z = z


    def rotation_x(self):
        return np.stack([[1, 0, 0],
                              [0, math.cos(math.radians(self.angle_x)), -math.sin(math.radians(self.angle_x))],
                              [0, math.sin(math.radians(self.angle_x)), math.cos(math.radians(self.angle_x))]])

    def rotation_y(self):
        return np.stack([[math.cos(math.radians(self.angle_y)), 0, -math.sin(math.radians(self.angle_y))],
                              [0, 1, 0],
                              [math.sin(math.radians(self.angle_y)), 0, math.cos(math.radians(self.angle_y))]])


    def rotation_z(self):
        return np.stack([[math.cos(math.radians(self.angle_z)),-math.sin(math.radians(self.angle_z)), 0],
                              [math.sin(math.radians(self.angle_z)), math.cos(math.radians(self.angle_z)), 0],
                              [0, 0, 1]])


    def find_transform(self):
        self.R = self.rotation_z() @ self.rotation_y() @ self.rotation_x()
        self.T[0,3] = self.X * 0.001
        self.T[1,3] = self.Y * 0.001
        self.T[2,3] = self.Z * 0.001
        self.T[3,3] = 1
        self.T[:3,:3] = self.R

        return self.T

class motions:
    def __init__(self, radius, center_x, center_y, number_points):
        self.radius = radius
        self.center_x = center_x
        self.center_y = center_y
        self.points_num = number_points
        self.circle = None

    def get_circlular_points(self):
        # random angle
        alpha = (2 * math.pi) / self.points_num
        # random radius
        r = self.radius
        points = list()
        angle = 0
        for i in range(self.points_num):
            # calculating coordinates
            angle = angle + alpha
            x = round((r * math.cos(angle) + self.center_x), 3)
            y = round((r * math.sin(angle) + self.center_y), 3)
            point = (x, y)
            points.append(point)

        return points



if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    client = Client("opc.tcp://localhost:5000/")

    try:
        client.connect()
        client.load_type_definitions()
        root = client.get_root_node()
        objects = client.get_objects_node()
        uri = "http://iiwa-control.ciirc.cz"
        idx = client.get_namespace_index(uri)
        rootObj = objects.get_child(["IIWA"])
        input = rootObj.get_child(["RobotGenericInput"])
        output = rootObj.get_child(["RobotGenericOutput"])
        # 'A': -90.001, 'B': -0.006, 'C': 179.998, 'z': 400.025, 'y': -600.006, 'x': -180.049}
        motion = motions(radius=100, center_x= -100, center_y= -500, number_points=8)
        points = motion.get_circlular_points()
        # move_to_position('world')
        for point in points:
            move_to_position_with_points('world',X= point[0], Y= point[1], Z= 350, RA= -90, RB=0, RC=180 )
        tra1 = get_current_pos_base()

    finally:
        client.disconnect()