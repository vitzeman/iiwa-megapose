import os
import copy

import open3d as o3d



# SCALE = (1/3.09244982483194) * 1000 
SCALE = 0.35*1000 # FOR NERFACTO BY VARUN
METHOD = "nerfacto"
# method = "bakedsdf"

ROOT_PATH = "/home/testbed/Projects/mesh_manipulation/data/reconstruction/" + METHOD

# list directory
meshes = os.listdir(os.path.join(ROOT_PATH, "meshes_cleared"))
save_dir = os.path.join(ROOT_PATH, "meshes_cleared_scaled")
os.makedirs(save_dir, exist_ok=True)

for mesh in meshes:
    if mesh != "d05_axle_front":
        continue
    mesh_path = os.path.join(ROOT_PATH, "meshes_cleared", mesh, "texture", "mesh.obj")
    print(mesh_path)
    original_mesh = o3d.io.read_triangle_mesh(
        mesh_path, True
    )  # True needs to be set to read texture

    # Scale the mesh
    scaled_mesh = copy.deepcopy(original_mesh)
    print(scaled_mesh.get_center())
    scaled_mesh.scale(SCALE, center=(0, 0, 0))

    # Save the mesh
    os.makedirs(os.path.join(save_dir, mesh), exist_ok=True)
    new_mesh_path = os.path.join(save_dir, mesh, "mesh.obj")
    o3d.io.write_triangle_mesh(new_mesh_path, scaled_mesh)

    # Visualize
    o3d.visualization.draw_geometries([original_mesh, scaled_mesh])
