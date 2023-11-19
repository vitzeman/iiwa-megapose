import numpy as np
from scipy.spatial.transform import Rotation as R


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
