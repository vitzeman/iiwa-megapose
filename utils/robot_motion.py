import math, time
from Helper.rotation_helper import t_matrix
import asyncio

class path:
    def __init__(self,center_x, center_y, number_points, distance, min_height, height_change, z_offset, max_height):
        """
        distance is the length from camera center point to world origin
        center_x and center_y and z_offset is the center of the hemisphere in x,y and z plane
        height_change is the sampling rate of the hemisphere
        max_height to set if you dont want to have points to the heights possible
        min_height to start sampling at height or z
        number_points is the number of points you want to sample on the circle
         """

        self.radius = None
        self.center_x = center_x
        self.center_y = center_y
        self.points_num = number_points
        self.height = None
        self.min_height = min_height
        self.angle = None
        self.height_change = height_change
        self.distance = distance
        self.z_offset = z_offset
        self.max_height = max_height

    # find the sampled heights and the radis of hemisphere
    def _find_height_and_radius(self):
        range_ofpoint = list()
        # for height in range(self.min_height, int((self.distance) ), self.height_change):
        for height in range(self.min_height, int((self.max_height) ), self.height_change):
            radius = int(math.sqrt((self.distance)**2 - (height)**2))
            one_loop = [height, radius]
            range_ofpoint.append(one_loop)
        return range_ofpoint


    def get_circlular_points(self ):
        points = list()
        num_of_point = self.points_num
        range_of_points = self._find_height_and_radius()

        for length in range_of_points:
            self.height = length[0]
            self.radius = length[1]
            self.angle = math.degrees(math.pi/2) - math.degrees(math.atan(self.height / self.radius))
            r = self.radius
            alpha = (2 * math.pi) / num_of_point

            # num_point = int(self.radius / 8)
            num_point = num_of_point
            print(" number of points are", num_point)

            angle = 0
            for i in range(num_point):
                # calculating coordinates
                angle = angle + alpha
                RB = self.angle * math.cos(angle)
                RC = self.angle * math.sin(angle)
                x = round((r * math.cos(angle) + self.center_x), 3)
                y = round((r * math.sin(angle) + self.center_y), 3)
                point = [x, y, RB, RC, self.height + self.z_offset ]
                points.append(point)
                print(point)

            # if (num_of_point % 2) == 0:
            #     num_of_point = self.points_num
            # else:
            #     num_of_point = self.points_num
        return points


class SubHandler(object):
    def __init__(self):
        self.RX = None
        self.RY = None
        self.RZ = None
        self.RA = None
        self.RB = None
        self.RC = None
        self.tranf_dict = {'ActualX': None, 'ActualY': None,
                           'ActualZ':None,'ActualRotZ':None,
                           'ActualRotY':None,'ActualRotX':None,
                           'OperationStarted': None, 'OperationFinished': None,'Failed': None}
        self.data_change = False
        self.inmotion = False
        self.fail = True
        self.event_change = False

    def datachange_notification(self, node, val, data):
        # check for change in OperationFinished
        self.tranf_dict[node.nodeid.Identifier] = val
        self.data_change = True

    def event_notification(self, event):
        if event.SourceNode.Identifier == "InMotionEvent":
            if event.Message.Text == 'ok':
                self.inmotion = False
            else:
                self.inmotion = True

        if event.SourceNode.Identifier == "FailEvent":
            if event.Message.Text == 'ok':
                self.fail = False
            else:
                self.fail = True

    def move_to_position_with_points(self,output, X, Y, Z, RA, RB, RC):
        print("sending robot to position")

        while self.inmotion:
            print("robot is in motion")
            time.sleep(1.0)

        output.get_child(["DataReady"]).set_value(False)
        output.get_child(["TargetX"]).set_value(X)
        output.get_child(["TargetY"]).set_value(Y)
        output.get_child(["TargetZ"]).set_value(Z)
        output.get_child(["TargetRA"]).set_value(math.radians(RA))
        output.get_child(["TargetRB"]).set_value(math.radians(RB))
        output.get_child(["TargetRC"]).set_value(math.radians(RC))
        output.get_child(["ProgramNumber"]).set_value(2)
        time.sleep(1.0)

        output.get_child(["DataReady"]).set_value(True)
        time.sleep(1.0)

        while not self.data_change:
            time.sleep(1.0)
            print("waiting for data to change - move to position")

        while not self.tranf_dict["OperationFinished"]:
            print("waiting for last operation to finish")
            time.sleep(1.0)

    def move_to_home(self, output):

        print("sending robot to position")

        while self.inmotion:
            print("robot is in motion")
            time.sleep(1.0)

        output.get_child(["DataReady"]).set_value(False)
        output.get_child(["TargetX"]).set_value(home['x'])
        output.get_child(["TargetY"]).set_value(home['y'])
        output.get_child(["TargetZ"]).set_value(home['z'])
        output.get_child(["TargetRA"]).set_value(math.radians(home['A']))
        output.get_child(["TargetRB"]).set_value(math.radians(home['B']))
        output.get_child(["TargetRC"]).set_value(math.radians(home['C']))
        output.get_child(["ProgramNumber"]).set_value(2)
        time.sleep(1.0)

        output.get_child(["DataReady"]).set_value(True)
        time.sleep(1.0)

        while not self.data_change:
            time.sleep(1.0)
            print("waiting for data to change - home")

        while not self.tranf_dict["OperationFinished"]:
            print("waiting for last operation to finish")
            time.sleep(1.0)


    def check_point_fail_pass(self, input):

        print("checking if last postion passed or failed ")
        self.data_change = False
        input.get_child(["DataReady"]).set_value(False)
        input.get_child(["ProgramNumber"]).set_value(3)
        time.sleep(1.0)
        input.get_child(["DataReady"]).set_value(True)

        while not self.data_change:
            print(self.data_change)
            time.sleep(1.0)
            print("waiting for data to change - check point")

        while not self.tranf_dict["OperationFinished"]:
            print("waiting for last operation to finish")
            time.sleep(1.0)

        print(self.tranf_dict["Failed"])
        if self.fail:
            print("failed to reach point")
            self.data_change = True
            return False
        return True


    def get_current_pos_base(self,input):
        print('Getting current position')

        while self.inmotion:
            print("robot is in motion")
            time.sleep(1.0)

        self.data_change = False
        input.get_child(["DataReady"]).set_value(False)
        input.get_child(["ProgramNumber"]).set_value(1)
        time.sleep(1.0)
        input.get_child(["DataReady"]).set_value(True)

        while not self.data_change:
            time.sleep(1.0)
            print("waiting for data to change- get current pos")

        while not self.tranf_dict["OperationFinished"]:
            print("waiting for last operation to finish")
            time.sleep(1.0)


        RX = self.tranf_dict["ActualX"]
        RY = self.tranf_dict["ActualY"]
        RZ = self.tranf_dict["ActualZ"]
        RA = self.tranf_dict["ActualRotZ"]
        RB = self.tranf_dict["ActualRotY"]
        RC = self.tranf_dict["ActualRotX"]
        input.get_child(["DataReady"]).set_value(True)
        print("RX ", round(float(RX), 3), "RY ", round(float(RY), 3), "RZ ", round(float(RZ), 3), "RA ",
              round(float(RA), 3), "RB ", round(float(RB), 3), "RC ", round(float(RC), 3))

        matrix = t_matrix()
        matrix.__int__(x=RX, y=RY, z=RZ, ra=RA, rb=RB, rc=RC)
        transformation = matrix.find_transform()
        self.data_change = False

        return transformation, RX, RY,RZ, RA, RB, RC


    def get_angle_and_translation(self,input):
        input.get_child(["ProgramNumber"]).set_value(1)
        input.get_child(["DataReady"]).set_value(False)
        time.sleep(1.0)

        # while input.get_child(["inMotion"]).get_value():
        #     time.sleep(1.0)
        RX = input.get_child(["ActualX"]).get_value()
        RY = input.get_child(["ActualY"]).get_value()
        RZ = input.get_child(["ActualZ"]).get_value()
        RA = input.get_child(["ActualRA"]).get_value()
        RB = input.get_child(["ActualRB"]).get_value()
        RC = input.get_child(["ActualRC"]).get_value()
        input.get_child(["DataReady"]).set_value(True)
        return RX, RY, RZ, RA, RB, RC

def get_angle_and_translation(input):
    input.get_child(["ProgramNumber"]).set_value(1)
    input.get_child(["DataReady"]).set_value(False)
    time.sleep(1.0)

    # while input.get_child(["inMotion"]).get_value():
    #     time.sleep(1.0)
    RX = input.get_child(["ActualX"]).get_value()
    RY = input.get_child(["ActualY"]).get_value()
    RZ = input.get_child(["ActualZ"]).get_value()
    RA = input.get_child(["ActualRA"]).get_value()
    RB = input.get_child(["ActualRB"]).get_value()
    RC = input.get_child(["ActualRC"]).get_value()
    input.get_child(["DataReady"]).set_value(True)
    return RX, RY, RZ, RA, RB, RC