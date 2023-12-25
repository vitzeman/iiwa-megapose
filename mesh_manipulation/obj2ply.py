import os

import numpy as np
import open3d as o3d


def obj2ply(obj_path: str, ply_path: str) -> None:
    """Converts a .obj file to a .ply file

    Args:
        obj_path (str): Path to the .obj file
        ply_path (str): Path to the .ply file
    """
    mesh = o3d.io.read_triangle_mesh(obj_path)
    o3d.io.write_triangle_mesh(ply_path, mesh)
    return None


if __name__ == "__main__":
    ply_path = "/home/vit/Documents/DP/megapose6d-cluster/6D_pose_dataset/BOP_format/Tags/models"
    obj_path = "/home/vit/Documents/DP/megapose6d-cluster/local_data/rc_car/meshes_CAD"
    contents = sorted(os.listdir(obj_path))
    for dir in contents:
        idx = dir.split("_")[0].replace("d", "").zfill(6)
        name_mesh = "obj_" + idx
        obj_path_mesh = os.path.join(obj_path, dir, "mesh.obj")
        ply_path_mesh = os.path.join(ply_path, name_mesh + ".ply")
        obj2ply(obj_path_mesh, ply_path_mesh)
