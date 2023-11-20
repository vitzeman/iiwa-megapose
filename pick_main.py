# 1st party
import argparse
import os
import time
import json
from typing import Any, Tuple, Union
import copy
import requests

# 3rd party
from mlsocket import MLSocket
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Mine
from cluster.Client import get_megapose_estimation
from camera.basler_camera import BaslerCamera
from camera.Frame_processor import FrameProcessor

LABELS = {
    1: "d01_controller",
    2: "d02_servo",
    3: "d03_main",
    4: "d04_motor",
    5: "d05_axle_front",
    6: "d06_battery",
    7: "d07_axle_rear",
    8: "d08_chassis",
}
LABELS_NUMS_KEY = [x + 48 for x in LABELS.keys()]  # ASCII code for numbers from 1 to 8


def parse_args() -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Pick and place")
    parser.add_argument("-h", "--host", type=str, default="10.35.129.250")
    parser.add_argument("-p", "--port", type=int, default=65432)

    return parser.parse_args()


# TODO: Maybe move each class to file of its own
# Frame_proccesing replaced with FrameProcessor
class Frame_processing(BaslerCamera):
    def __init__(
        self,
        serial_number: str = "24380112",
        camera_parametes: str = os.path.join("camera", "camera_parameters.json"),
        save_location: str = "",
    ) -> None:
        super().__init__(serial_number, camera_parametes, save_location)
        self.camera_ideal_params = self.get_ideal_camera_parameters()

        self.frame = None
        self.bbox = None
        self.idx = None

        self.window_name = "Proccess frame"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.extract_coordinates)

    def reset(self):
        self.frame = None
        self.bbox = None
        self.idx = None

    def extract_coordinates(self, event, x, y, flags, parameters):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print("Left click")
            self.bbox = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            # print("Left release")
            self.bbox.extend([x, y])
            self.bbox = np.array(self.bbox)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # print("Right click")
            self.bbox = None

    def proccess_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: undistor the image

        frame = self.get_single_image()
        frame = self.undistort_image(frame)
        frame_vis = copy.deepcopy(frame)
        key = cv2.waitKey(1) & 0xFF
        should_quit = False
        if key == ord("q"):
            should_quit = True
        elif key == ord("h"):  # help
            print("q - quit NOT IMPLEMENTED")
            print("h - help")
            print("To select object press number from 1 to 8")
            for key, value in LABELS.items():
                print(f"  {value} - {key}")
            print("To set bbox click on the image and drag the mouse - NOT IMPLEMENTED")
            print("To confirm press Enter")
            print("To reset press r")
        elif key in LABELS_NUMS_KEY:
            self.idx = np.array([key - 48])  # Convert ASCII code to number
            print(f"Selected object: {LABELS[self.idx[0]]}")
        elif key == ord("r"):
            self.reset()
        elif key == 13:  # enter
            self.frame = frame
            # TODO: does not work for some reason
            frame_vis = cv2.addWeighted(
                frame_vis, 0.5, np.zeros_like(frame_vis), 0.5, 0
            )
            frame_vis = cv2.putText(
                frame_vis,
                f"Running_inference on {LABELS[self.idx[0]]}",
                (400, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if self.idx is None:
            cv2.putText(
                frame_vis,
                "Selected object: -",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame_vis,
                f"Selected object: {LABELS[self.idx[0]]}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if self.bbox is not None and len(self.bbox) == 4:
            cv2.rectangle(
                frame_vis,
                (self.bbox[0], self.bbox[1]),
                (self.bbox[2], self.bbox[3]),
                (0, 255, 0),
                2,
            )

        cv2.imshow(self.window_name, frame_vis)

        return should_quit, self.frame, self.bbox, self.idx


class IIWA_tools:
    def __init__(
        self, TX: float, TY: float, TZ: float, A: float, B: float, C: float, name: str
    ):
        """IIWA tools class to define the tools that are attached to the robot flange
        rotatation matrix is calculated using zyx euler angles

        Args:
            TX (float): x translation in mm
            TY (float): y translation in mm
            TZ (float): z translation in mm
            A (float): rotation around z axis in degrees
            B (float): rotation around y axis in degrees
            C (float): rotation around x axis in degrees
            name (str): Name of the tool
        """
        self.TX = TX  # mm
        self.TY = TY  # mm
        self.TZ = TZ  # mm
        self.A = A
        self.B = B
        self.C = C
        self.name = name
        # Why is this here?
        self.calculateframe()

    def __repr__(self) -> str:
        return f"Tool: {self.name}, TX: {self.TX}, TY: {self.TY}, TZ: {self.TZ}, A: {self.A}, B: {self.B}, C: {self.C}"

    def __str__(self):
        return self.name

    def calculateframe(self) -> np.ndarray:
        """#TODO : Docstring for calculateframe.

        Returns:
            np.ndarray: Transformation matrix of the tool
        """
        RotationMatrix = R.from_euler("zyx", [self.A, self.B, self.C], degrees=True)
        TranslationMatrix = np.array([self.TX, self.TY, self.TZ]).reshape(
            3,
        )
        TransformationMatrix = np.eye(4)

        TransformationMatrix[:3, :3] = RotationMatrix.as_matrix()
        TransformationMatrix[:3, 3] = TranslationMatrix
        return TransformationMatrix


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
        print("closing the gripper")
        time.sleep(1.0)

    def openGripper(self):
        while self.checkReady() != "OK":
            pass

        print("opening the gripper")
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

        X, Y, Z, A, B, C = self.finddestinationframe(X, Y, Z, A, B, C, tool)

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


def frame_processing_test():
    """Test function for FrameProcessor class"""
    detector = FrameProcessor()
    detector.connect()
    detector.adjust_camera()

    while True:
        should_quit, frame, bbox, idx = detector.proccess_frame()
        if should_quit:
            break
        if frame is not None or bbox is not None or idx is not None:
            time.sleep(5)

    cv2.destroyAllWindows()
    detector.disconnect()


def movement_test():
    """Test function for movement of the robot"""
    recorded_path = os.path.join("pose_data", "d03_main.json")
    with open(recorded_path, "r") as f:
        data = json.load(f)

    print(data)

    pose = np.array(data["pose"])
    print(pose)

    robot1_ip = "172.31.1.10"
    robot1 = IIWA(robot1_ip)

    # create the tools
    # camera = IIWA_tools(TX=50, TY=0, TZ=0, A=0, B=0, C=0, name='camera')
    # gripper = IIWA_tools(TX=0, TY=0, TZ=200, A=0, B=0, C=0, name='gripper')

    iiwa_camera = IIWA_tools(TX=3, TY=-90, TZ=-15, A=0, B=0, C=0, name="camera")
    iiwa_gripper = IIWA_tools(
        TX=15, TY=0, TZ=230, A=0, B=0, C=0, name="gripper"
    )  # Tz = 230 for now 226 for touching the table

    # attach tool to the flange of the robot
    robot1.addTool(iiwa_camera)
    robot1.addTool(iiwa_gripper)

    robot1.closeGripper(position=0)
    robot1.sendCartisianPosition(
        X=-200, Y=-500, Z=300, A=0, B=0, C=180, motion="ptp", tool=iiwa_gripper
    )
    robot1.sendCartisianPosition(
        X=-200, Y=-500, Z=300, A=0, B=0, C=180, motion="ptp", tool= iiwa_camera
    )

    robot1.wait2ready()

    # robot1.sendCartisianPosition(X=-200, Y=-500, Z=300, A=90, B=0, C=180, motion='ptp', tool=gripper)
    # robot1.sendCartisianPosition(X=-200, Y=-500, Z=300, A=180, B=0, C=180, motion='ptp', tool=gripper)
    # robot1.openGripper()
    # robot1.sendCartisianPosition(X=-200, Y=-500, Z=50, A=180, B=0, C=180, motion='ptp', tool=gripper)
    # robot1.closeGripper(position=23000)
    # robot1.sendCartisianPosition(X=-200, Y=-500, Z=300, A=90, B=0, C=180, motion='ptp', tool=gripper)
    # robot1.wait2ready() # THis will ensure that the robot is ready to go to the next position


# THIS SHOULD BE THE MAIN FUNCTION
def main():
    # Camera init
    # detector = Frame_processing()
    detector = FrameProcessor()
    detector.connect()
    detector.adjust_camera()
    detector.camera

    # TODO: Add parser for the host and port
    # Server comunication init
    ml_socket = MLSocket()
    host = "10.35.129.250"
    port = 65432
    ml_socket.connect((host, port))
    print(f"Connection to {host}:{port} established")

    # Main loop
    while True:
        should_quit, frame, bbox, idx = detector.proccess_frame()
        if should_quit:  # Should quit after q is pressed in the window
            # This will sent data to server to stop the server and close the connection (idx = -1)
            _ = get_megapose_estimation(
                ml_socket, np.zeros((3, 3, 3)), np.zeros((4)), np.array([-1])
            )
            break

        if frame is None or bbox is None or idx is None:
            continue

        if (
            type(frame) != np.ndarray
            or type(bbox) != np.ndarray
            or type(idx) != np.ndarray
        ):
            print("Wrong type")
            continue

        # Get the pose from megapose running on the cluster
        pose = get_megapose_estimation(ml_socket, frame, bbox, idx)

        # Saving the data
        print(f"Data received: {pose}")
        recorded_data = {
            "labl": LABELS[idx[0]],
            "bbox": bbox.tolist(),
            "idx": idx.tolist(),
            "pose": pose.tolist(),
        }
        cv2.imwrite(os.path.join("pose_data", f"{LABELS[idx[0]]}.png"), frame)
        with open(os.path.join("pose_data", f"{LABELS[idx[0]]}.json"), "w") as f:
            json.dump(recorded_data, f, indent=2)

        detector.reset()

        # TODO: Plan the movement of the robot based on the pose

    # Deactivate everything else
    # ml_socket.close()
    detector.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # main()
    # frame_processing_test()
    movement_test()
