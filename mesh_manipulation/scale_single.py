import os
import copy

import open3d as o3d

scale = 0.633505210660958 * 1000

path2mesh = "/home/vit/Documents/DP/iiwa-megapose/mesh_manipulation/data/Scenes"
meshpath = os.path.join(path2mesh, "mesh_cropped_40.obj")
original_mesh = o3d.io.read_triangle_mesh(
    meshpath, True
)  # True needs to be set to read texture
scaled_mesh = copy.deepcopy(original_mesh)
print("loaded")

scaled_mesh.scale(scale, center=(0, 0, 0))
print("scaled")
save_path = os.path.join(path2mesh, "mesh_cropped_40_scaled.obj")
o3d.io.write_triangle_mesh(save_path, scaled_mesh, write_triangle_uvs=True)
