import os
import copy

import open3d as o3d
import numpy as np

scale = 0.633505210660958 * 1000

path2mesh = "/home/vit/Documents/DP/iiwa-megapose/mesh_manipulation/data/Scenes"
meshpath = os.path.join(path2mesh, "mesh_cropped_40_scaled.obj")
original_mesh = o3d.io.read_triangle_mesh(
    meshpath, True
)  # True needs to be set to read texture
scaled_mesh = copy.deepcopy(original_mesh)
print("loaded")

min_bound = scaled_mesh.get_min_bound()
max_bound = scaled_mesh.get_max_bound()

min_bound[1] = 0

scaled_mesh.crop([min_bound, max_bound])

save_path = os.path.join(path2mesh, "mesh_cropped_40_scaled_cropped.obj")
o3d.io.write_triangle_mesh(save_path, scaled_mesh, write_triangle_uvs=True)
