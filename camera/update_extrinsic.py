import os
import json

import numpy as np
from tqdm.contrib import tenumerate, tzip


if __name__ == "__main__":
    with open(
        os.path.join("camera", "data", "Benchmark_scene1", "transforms.json")
    ) as f:
        aruco_dict = json.load(f)

    # save aruco_dict
    with open(
        os.path.join("camera", "data", "Benchmark_scene1", "transforms_aruco.json"), "w"
    ) as f:
        json.dump(aruco_dict, f, indent=2)

    with open(
        os.path.join("camera", "data", "Benchmark_scene1", "base_extrinsics.json")
    ) as f:
        refined_poses = json.load(f)

    aruco_frames = aruco_dict["frames"]

    for a_frame, r_frame in tzip(aruco_frames, refined_poses):
        T_mtx_refined = np.array(r_frame["transform_matrix"])
        T_mtx_refined = np.vstack((T_mtx_refined, np.array([0, 0, 0, 1])))

        a_frame["transform_matrix"] = T_mtx_refined.tolist()

    aruco_dict["orientation_override"] = "none"
    aruco_dict["applied_scale"] = 1.0
    aruco_dict["applied_transform"] = np.eye(4)[:3, :].tolist()

    # save aruco_dict
    with open(
        os.path.join("camera", "data", "Benschmark_scene1", "transforms.json"),
        "w",
    ) as f:
        json.dump(aruco_dict, f, indent=2)
