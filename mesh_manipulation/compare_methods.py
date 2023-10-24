import open3d as o3d
import numpy as np
import copy
import os



# First folder 
ROOT_PATH = "/home/testbed/Projects/mesh_manipulation/data/reconstruction/"

first_method = "nerfacto"
second_method = "bakedsdf"

first_meshes = os.listdir(os.path.join(ROOT_PATH, first_method, "meshes_cleared_scaled"))
second_meshes = os.listdir(os.path.join(ROOT_PATH, second_method, "meshes_cleared_scaled"))

for mesh in second_meshes:
    # Load the meshes
    first_mesh_path = os.path.join(ROOT_PATH, first_method, "meshes_cleared_scaled", mesh, "mesh.obj")
    second_mesh_path = os.path.join(ROOT_PATH, second_method, "meshes_cleared_scaled", mesh, "mesh.obj")
    first_mesh = o3d.io.read_triangle_mesh(first_mesh_path, True)
    second_mesh = o3d.io.read_triangle_mesh(second_mesh_path, True)

    # Visualize
    o3d.visualization.draw_geometries([first_mesh, second_mesh])

