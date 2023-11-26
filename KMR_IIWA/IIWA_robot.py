# Standrat library
import json
import time
import requests
from typing import Tuple

# Third party
import numpy as np
from scipy.spatial.transform import Rotation as R


# Mine
from KMR_IIWA.IIWA_tools import IIWA_tools


class IIWA:
    """IIWA class to control the robot KMR troght http requests to the robot"""

    def __init__(self, ip):
        self.ip = ip
        self._get_cartisian_postion = (
            "http://" + self.ip + ":30000/" + "GetCartesianposition"
        )
        self._get_joint_positon = "http://" + self.ip + ":30000/" + "GetJointpostion"
        self._send_cartiesian_position = (
            "http://" + self.ip + ":30000/" + "GotoCartesianposition"
        )
        self._get_gripper_data = "http://" + self.ip + ":30000/" + "GetGripperpostion"
        self._senf_joint_position = (
            "http://" + self.ip + ":30000/" + "GotoJointposition"
        )
        self._close_griper = "http://" + self.ip + ":30000/" + "CloseGripper"
        self._open_gripper = "http://" + self.ip + ":30000/" + "OpenGripper"
        self._ready_to_send = "http://" + self.ip + ":30000/" + "ready"
        self._last_operation = "http://" + self.ip + ":30000/" + "failed"
        self.tool = {}

    def addTool(self, tool: IIWA_tools):
        """Add tool to the robot

        Args:
            tool (IIWA_tools): Tool to be added
        """
        self.tool[tool.name] = tool.calculateframe()

    def checkReady(self):
        """Check if the robot is ready to receive new commands

        Returns:
            _type_: _description_
        """
        result = requests.get(url=self._ready_to_send)
        time.sleep(1.0)
        result = result.content.decode()
        return result

    def checkLastOperation(self):
        result = requests.get(url=self._last_operation)
        time.sleep(1.0)
        result = result.content.decode()
        return result

    def getGripperpos(self):
        result = requests.get(url=self._get_gripper_data)
        time.sleep(1.0)
        result = result.content.decode()
        return result

    def closeGripper(self, position=40000, speed=40000, force=25000):
        while self.checkReady() != "OK":
            pass
        params = (
            "&position="
            + str(position)
            + "&speed="
            + str(speed)
            + "&force="
            + str(force)
        )
        close_operation = self._close_griper + "/?" + params
        requests.get(url=close_operation)
        print(
            f"Closing gripper to position {position} with speed {speed} and force {force}"
        )
        time.sleep(1.0)

    def openGripper(self):
        while self.checkReady() != "OK":
            pass

        print("Opening the gripper")
        requests.get(url=self._open_gripper)
        time.sleep(1.0)

    def getCartisianPosition(
        self, tool: IIWA_tools = None, degree: bool = True, stop: bool = True
    ) -> dict:
        """#TODO : Docstring for getCartisianPosition.

        Args:
            tool (IIWA_tools, optional): _description_. Defaults to None.
            degree (bool, optional): _description_. Defaults to True.
            stop (bool, optional): _description_. Defaults to True.

        Returns:
            dict: _description_
        """
        if stop:
            while self.checkReady() != "OK":
                pass

        result = requests.get(url=self._get_cartisian_postion)
        time.sleep(1.0)
        position = json.loads(result.content)

        transformTool = None
        if tool:
            transformTool = self.tool[tool.name]
            position["tool"] = tool.name
        else:
            transformTool = np.eye(4)
            position["tool"] = "flange"

        RotationMatrix = R.from_euler(
            "zyx", [position["A"], position["B"], position["C"]], degrees=False
        )
        TranslationMatrix = np.array(
            [position["x"], position["y"], position["z"]]
        ).reshape(
            3,
        )

        TransformationMatrixflange = np.eye(4)
        TransformationMatrixflange[:3, :3] = RotationMatrix.as_matrix()
        TransformationMatrixflange[:3, 3] = TranslationMatrix

        Transformation = transformTool @ TransformationMatrixflange

        Rot = Transformation[:3, :3]
        Tra = Transformation[:3, 3]

        angles = R.from_matrix(Rot).as_euler("zyx", degrees=False)
        position["A"] = angles[0]
        position["B"] = angles[1]
        position["C"] = angles[2]
        position["x"] = Tra[0]
        position["y"] = Tra[1]
        position["z"] = Tra[2]

        if degree:
            position["A"] = np.rad2deg(position["A"])
            position["B"] = np.rad2deg(position["B"])
            position["C"] = np.rad2deg(position["C"])

        return position

    def getJointPostion(self, degree: bool = True):
        result = requests.get(url=self._get_joint_positon)
        time.sleep(1.0)
        position = json.loads(result.content)

        if degree:
            position["A1"] = np.rad2deg(position["A1"])
            position["A2"] = np.rad2deg(position["A2"])
            position["A3"] = np.rad2deg(position["A3"])
            position["A4"] = np.rad2deg(position["A4"])
            position["A5"] = np.rad2deg(position["A5"])
            position["A6"] = np.rad2deg(position["A6"])
            position["A7"] = np.rad2deg(position["A7"])

        return position

    def finddestinationframe(self, X, Y, Z, A, B, C, tool):
        TransformationMatrix = np.eye(4)
        RotationMatrix = R.from_euler("zyx", [A, B, C], degrees=True)
        TranslationMatrix = np.array([X, Y, Z]).reshape(
            3,
        )
        TransformationMatrix[:3, :3] = RotationMatrix.as_matrix()
        TransformationMatrix[:3, 3] = TranslationMatrix

        if tool:
            transformTool = self.tool[tool.name]
        else:
            transformTool = np.eye(4)

        destination = transformTool @ TransformationMatrix

        destRot = destination[:3, :3]
        destTra = destination[:3, 3]

        result = R.from_matrix(destRot).as_euler("zyx", degrees=True)
        A = result[0]
        B = result[1]
        C = result[2]
        X = destTra[0]
        Y = destTra[1]
        Z = destTra[2]

        return X, Y, Z, A, B, C
    

    def get_pose_of_tool(self, X, Y, Z, A, B, C, tool):
        if not tool: # If flange is supposed to be used nothing changes
            return X, Y, Z, A, B, C
        

        T_T2F = tool.T_T2F
        T = np.eye(4)
        T[:3, :3] = R.from_euler("ZYX", [A, B, C], degrees=True).as_matrix()
        T[:3, 3] = np.array([X, Y, Z])

        T_fin = T @ T_T2F
        nX, nY, nZ = T_fin[:3, 3]
        nA, nB, nC = R.from_matrix(T_fin[:3, :3]).as_euler("ZYX", degrees=True)

        return nX, nY, nZ, nA, nB, nC


    def distancecurdest(self, X, Y, Z, tool):
        cartpostion = self.getCartisianPosition(tool)
        time.sleep(1.0)
        current = np.array(([cartpostion["x"], cartpostion["y"], cartpostion["z"]]))
        dest = np.array((X, Y, Z))
        distance = np.linalg.norm(current - dest)
        return distance

    def wait2ready(self) -> None:
        """Wait until the robot is ready that is mainly waiting for the robot to stop moving"""
        while self.checkReady() != "OK":
            pass
        print("Ready to go")

    def sendCartisianPosition(
        self,
        X: float,
        Y: float,
        Z: float,
        A: float,
        B: float,
        C: float,
        motion: str = "lin",
        speed: float = 0.1,
        tool: IIWA_tools = None,
        degree: bool = True,
        desc: str = None,
    ) -> str:
        """_summary_

        Args:
            X (float): _description_
            Y (float): _description_
            Z (float): _description_
            A (float): _description_
            B (float): _description_
            C (float): _description_
            motion (str, optional): _description_. Defaults to 'lin'.
            speed (float, optional): _description_. Defaults to 0.1.
            tool (IIWA_tools, optional): _description_. Defaults to None.
            degree (bool, optional): _description_. Defaults to True.
            desc (str, optional): Additional descriptin. Defaults to None.

        Returns:
            str: _description_
        """
        while self.checkReady() != "OK":
            pass

        if desc:
            print(desc)

        print(
            f"Sending cartisian position: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}, A={A:.2f}, B={B:.2f}, C={C:.2f}, tool={tool}, motion={motion}, speed={speed}"
        )

        # X, Y, Z, A, B, C = self.finddestinationframe(X, Y, Z, A, B, C, tool)
        X, Y, Z, A, B, C = self.get_pose_of_tool(X, Y, Z, A, B, C, tool)

        tra = "TX=" + str(X) + "&TY=" + str(Y) + "&TZ=" + str(Z)
        if degree:
            rot = (
                "&TA="
                + str(np.deg2rad(A))
                + "&TB="
                + str(np.deg2rad(B))
                + "&TC="
                + str(np.deg2rad(C))
            )

        motion_movement = "&Motion=" + motion
        robot_speed = "&Speed=" + str(speed)
        s = tra + rot + motion_movement + robot_speed

        send_operation = self._send_cartiesian_position + "/?" + s
        requests.get(send_operation)
        time.sleep(1.0)

        last_operation = self.checkLastOperation()
        if last_operation == "OK":
            return "Going to position"
        else:
            return "Cant go to the specified place"
