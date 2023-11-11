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


class CameraPoseVisualizer:
    def __init__(
        self, xlim: tuple = (-1, 1), ylim: tuple = (-1, 1), zlim: tuple = (-1, 1)
    ) -> None:
        self.fig = plt.figure(figsize=(7, 7))
        # self.ax = self.fig.gca(projection="3d")
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        length = min(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]) * 0.1

        self.ax.quiver(0, 0, 0, 1, 0, 0, length=length, color="r", linewidth=1)
        self.ax.quiver(0, 0, 0, 0, 1, 0, length=length, color="g", linewidth=1)
        self.ax.quiver(0, 0, 0, 0, 0, 1, length=length, color="b", linewidth=1)
        self.ax.text(0, 0, 0, "O", fontweight="bold")

    def add_camera(
        self,
        T_mtx: np.ndarray,
        color: str = "r",
        focal_len_scaled: float = 1,
        aspect_ratio: float = 0.4,
        text: str = None,
    ) -> None:
        # TODO: Add another types for pose not just 4x4 matrix
        # if isinstance(pose, R):
        #     pose = pose.as_matrix()

        # vertex_std = np.array(
        #     [
        #         [0, 0, 0, 1],
        #         [
        #             focal_len_scaled * aspect_ratio,
        #             -focal_len_scaled * aspect_ratio,
        #             focal_len_scaled,
        #             1,
        #         ],
        #         [
        #             focal_len_scaled * aspect_ratio,
        #             focal_len_scaled * aspect_ratio,
        #             focal_len_scaled,
        #             1,
        #         ],
        #         [
        #             -focal_len_scaled * aspect_ratio,
        #             focal_len_scaled * aspect_ratio,
        #             focal_len_scaled,
        #             1,
        #         ],
        #         [
        #             -focal_len_scaled * aspect_ratio,
        #             -focal_len_scaled * aspect_ratio,
        #             focal_len_scaled,
        #             1,
        #         ],
        #     ]
        # )

        vertex_std = np.array(
            [
                [0, 0, focal_len_scaled, 1],
                [
                    focal_len_scaled * aspect_ratio,
                    -focal_len_scaled * aspect_ratio,
                    0,
                    1,
                ],
                [
                    focal_len_scaled * aspect_ratio,
                    focal_len_scaled * aspect_ratio,
                    0,
                    1,
                ],
                [
                    -focal_len_scaled * aspect_ratio,
                    focal_len_scaled * aspect_ratio,
                    0,
                    1,
                ],
                [
                    -focal_len_scaled * aspect_ratio,
                    -focal_len_scaled * aspect_ratio,
                    0,
                    1,
                ],
            ]
        )

        vertex_std = vertex_std @ T_mtx.T
        meshes = [
            [
                vertex_std[0, :-1],
                vertex_std[1, :-1],
                vertex_std[2, :-1],
            ],
            [
                vertex_std[0, :-1],
                vertex_std[2, :-1],
                vertex_std[3, :-1],
            ],
            [
                vertex_std[0, :-1],
                vertex_std[3, :-1],
                vertex_std[4, :-1],
            ],
            [
                vertex_std[0, :-1],
                vertex_std[4, :-1],
                vertex_std[1, :-1],
            ],
            [
                vertex_std[1, :-1],
                vertex_std[2, :-1],
                vertex_std[3, :-1],
                vertex_std[4, :-1],
            ],
        ]
        self.ax.add_collection3d(
            Poly3DCollection(
                meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.0
            )
        )
        if text:
            self.ax.text(*vertex_std[0, :-1], text, color=color)

    def show(self):
        self.ax.legend()
        plt.show()

        # self.fig.show()

    def save(self, path: str):
        self.fig.savefig(path)

    def add_legend_entry(self, color: str, text: str):
        self.ax.scatter([], [], marker="v", color=color, label=text)


if __name__ == "__main__":
    # Testing_part
    # cam_pose_vis = CameraPoseVisualizer(xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 1))

    # cam_pose_vis.add_camera(np.eye(4), text="0")
    # T_mtx = np.eye(4)
    # T_mtx[:3, 3] = 1
    # cam_pose_vis.add_camera(T_mtx, color="b", text="1")
    # cam_pose_vis.show()
    # cam_pose_vis.save("test.png")
    cam_pose_vis = CameraPoseVisualizer(xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 1))
    with open(
        os.path.join("camera", "data", "Benchmark_scene1", "transforms.json")
    ) as f:
        aruco_dict = json.load(f)

    with open(
        os.path.join("camera", "data", "Benchmark_scene1", "base_extrinsics.json")
    ) as f:
        refined_poses = json.load(f)

    aruco_frames = aruco_dict["frames"]
    for e, a_frame in enumerate(
        tqdm(aruco_frames, desc="Processing aruco", unit="image")
    ):
        name = a_frame["file_path"].split("/")[-1].split(".")[0]
        num = int(name)
        name = str(num)
        T_mtx = np.array(a_frame["transform_matrix"])

        if e % 10 == 0:
            cam_pose_vis.add_camera(T_mtx, color="b", text=None, focal_len_scaled=0.1)

    cam_pose_vis.add_legend_entry("b", "Aruco")

    for e, r_frame in enumerate(
        tqdm(refined_poses, desc="Processing refined", unit="image")
    ):
        num = r_frame["id"]
        name = str(num)
        T_mtx = np.array(r_frame["transform_matrix"])
        T_mtx = np.vstack((T_mtx, np.array([0, 0, 0, 1])))

        if e % 10 == 0:
            cam_pose_vis.add_camera(T_mtx, color="r", text=None, focal_len_scaled=0.1)

    cam_pose_vis.add_legend_entry("r", "Refined")

    cam_pose_vis.show()
    cam_pose_vis.save("test.png")
