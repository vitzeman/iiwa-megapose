import os
import json
from typing import Tuple, Union

import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle


class TransformationVisualizer:
    def __init__(
        self,
        xlim: tuple = (-1, 1),
        ylim: tuple = (-1, 1),
        zlim: tuple = (-1, 1),
        units: str = "-",
    ) -> None:
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)

        self.ax.set_xlabel(f"x [{units}]")
        self.ax.set_ylabel(f"y [{units}]")
        self.ax.set_zlabel(f"z [{units}]")

        length = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]) * 0.1

        self.ax.quiver(0, 0, 0, 1, 0, 0, length=length, color="r", linewidth=1)
        self.ax.quiver(0, 0, 0, 0, 1, 0, length=length, color="g", linewidth=1)
        self.ax.quiver(0, 0, 0, 0, 0, 1, length=length, color="b", linewidth=1)
        self.ax.text(0, 0, 0, "O", fontweight="bold")

        # Points of the coordinate system in homogeneous coordinates
        self.O = np.array([0, 0, 0, 1])
        self.X = np.array([1, 0, 0, 1])
        self.Y = np.array([0, 1, 0, 1])
        self.Z = np.array([0, 0, 1, 1])

    def add_transformation(
        self,
        T_mtx: np.ndarray,
        text: str = None,
    ) -> None:
        if isinstance(T_mtx, R):
            T_mtx = T_mtx.as_matrix()

        assert T_mtx.shape == (4, 4), "Transformation matrix must be 4x4"

        # Compute new points of the coordinate system
        O = self.O @ T_mtx
        X = self.X @ T_mtx
        Y = self.Y @ T_mtx
        Z = self.Z @ T_mtx

        # Plot the coordinate system
        self.ax.quiver(
            O[0], O[1], O[2], X[0], X[1], X[2], length=1, color="r", linewidth=1
        )
        self.ax.quiver(
            O[0], O[1], O[2], Y[0], Y[1], Y[2], length=1, color="g", linewidth=1
        )
        self.ax.quiver(
            O[0], O[1], O[2], Z[0], Z[1], Z[2], length=1, color="b", linewidth=1
        )
        if text is not None:
            self.ax.text(O[0], O[1], O[2], text, fontweight="bold")

    def add_title(self, title: str) -> None:
        self.ax.set_title(title)

    def show(self) -> None:
        self.ax.legend()
        plt.show()

    def save(self, path: str) -> None:
        self.ax.legend()
        plt.savefig(path)

    def add_point(self, point: np.ndarray, color: str = "k", text: str = None) -> None:
        self.ax.scatter(point[0], point[1], point[2], color=color)
        if text is not None:
            self.ax.text(point[0], point[1], point[2], text, fontweight="bold")
