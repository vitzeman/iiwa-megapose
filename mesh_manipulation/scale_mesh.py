import os
import copy

import open3d as o3d

items = (
        "d01_controller",
        "d02_servo",
        "d03_main",
        "d04_motor",
        "d05_axle_front",  # NOT AVAILABLE
        "d06_battery",
        "d07_axle_rear",  # NOT AVAILABLE
        "d08_chassis",  # NOT AVAILABLE
    )  # possible items for allignment
scales = {
    "d01_controller":  3.09244982483194,
    "d02_servo": 3.0869115311413973,
    "d03_main": 3.0894086594032033,
    "d04_motor": 3.0898800594246807,
    "d05_axle_front": 3.0770583466173473,
    "d06_battery": 3.089705081253575,
    "d07_axle_rear": 3.079723557204711,
    "d08_chassis": 3.079275596756827,
}
transformations = {
    "d01_controller": [ -0.5312432646751404, 0.00018635109881870449,-0.6334784030914307],
    "d02_servo": [ -0.5320762395858765, 0.00018001801799982786,-0.6340343952178955] ,
    "d03_main":  [ -0.5316610336303711, 0.0002628962101880461,-0.6337578296661377],
    "d04_motor":  [-0.5316621661186218 , 0.000276119913905859,-0.6337534785270691],
    "d05_axle_front":  [ -0.533478319644928, 0.0001823628117563203, -0.5950324535369873],
    "d06_battery":  [ -0.5316640734672546, 0.00026901226374320686,-0.6337578892707825],
    "d07_axle_rear":  [ -0.5324374437332153, 0.00018274436297360808, -0.614810585975647],
    "d08_chassis":  [ -0.5324349403381348, 0.00018191740673501045,-0.6148137450218201],
}


selected_items = [4]
ROOT_PATH = "/home/testbed/Projects/mesh_manipulation/data/reconstruction/bakedsdf"

for idx in selected_items:
    item = items[idx]
    print(item)
    scale = 1/scales[item] * 1000
    # transformation = [-1* scales[item] * x for x in transformations[item]]
    transformation = [-1* scale / 1000 * x for x in transformations[item]]
    # print(transformation)
    mesh_path = os.path.join(ROOT_PATH, "meshes_cleared", item, "texture", "mesh.obj")

    print(mesh_path)
    original_mesh = o3d.io.read_triangle_mesh(
        mesh_path, True
    )  # True needs to be set to read texture
    scaled_mesh = copy.deepcopy(original_mesh)

    scaled_mesh.translate(transformation)
    scaled_mesh.scale(scale, center=(0, 0, 0))
    # Save the mesh
    save_path = os.path.join(ROOT_PATH, "meshes_cleared_scaled", item)
    os.makedirs(save_path, exist_ok=True)

    # o3d.visualization.draw_geometries([scaled_mesh])
    new_mesh_path = os.path.join(save_path, "mesh.obj")
    print(f"[INFO]: Saving mesh to {new_mesh_path}")
    o3d.io.write_triangle_mesh(new_mesh_path, scaled_mesh)
