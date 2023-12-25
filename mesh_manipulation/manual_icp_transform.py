import open3d as o3d
import numpy as np
import copy
import os
import json


COLORS = {
    "blue": (0.1, 0.1, 0.7),
    "green": (0.1, 0.7, 0.1),
    "red": (0.7, 0.1, 0.1),
    "cyan": (0.1, 0.7, 0.7),
    "magenta": (0.7, 0.1, 0.7),
    "yellow": (0.7, 0.7, 0.1),
}

# Constants
ROOT_PATH = "~/Documents/megapose"
cropped_meshes_name = "extended_non_transformed_scaled_clean_meshes_nerf"

# For some reason the meshes cannot start wuth a ~/ so use relative path
# nerfactor_mesh_path = "../megapose/extended_non_transformed_scaled_clean_meshes_nerf/nerfacto_all/e08_3d1/mesh.obj"
# cad_mesh_path = "../megapose/CADmodels/obj/Axle_Rear.obj"

# nerfactor_mesh_path = "../megapose/extended_non_transformed_scaled_clean_meshes_nerf/nerfacto_all/e09_3d2/mesh.obj"
# cad_mesh_path = "../megapose/CADmodels/obj/Chassis_assambled.obj"


def draw_registration_result(
    input: o3d.geometry.PointCloud,
    goal: o3d.geometry.PointCloud,
    transformation: np.ndarray,
) -> None:
    """Visualizes the 2 meshes given transformation

    Args:
        source (o3d.geometry.PointCloud): Input mesh to be aligned to the target mesh
        target (o3d.geometry.PointCloud): Target mesh
        transformation (np.ndarray): Transformation matrix which aligns the source to target
    """
    vis_input = copy.deepcopy(input)
    vis_goal = copy.deepcopy(goal)
    vis_input.paint_uniform_color(COLORS["blue"])
    vis_goal.paint_uniform_color(COLORS["green"])

    vis_input.transform(transformation)
    o3d.visualization.draw_geometries([vis_input, vis_goal])
    return None


def pick_points(pcd: o3d.geometry.PointCloud) -> list[int]:
    """Let you pick up ids of points from the pcd with the mouse and return them
        Usage:
            [shift + left click] to add point picking
            [shift + right click] to undo point picking
            [Q] or [Esc] key to close window

    Args:
        pcd (o3d.geometry.Pointcloud): poincloud should be in point cloud format not mesh

    Returns:
        list[int]: list of ids
    """
    print("[INFO]: Please pick at least three correspondences")
    print("[INFO]: Press [shift + left click] to add point picking")
    print("[INFO]: Press [shift + right click] to undo point picking")
    print("[INFO]: Close window once done by pressing [Q] or [Esc] key")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_picked_points()


def manual_icp(goal_path: str, input_path: str) -> None:  # Fill here what to return
    """Aligns input mesh to the goal mesh using manualy selected points.

    Args:
        goal_path (str): Path to the goal mesh
        input_path (str): Path to the input mesh
    """
    # Load the meshe with open3d with problems with ASSIMP
    goal_mesh = o3d.io.read_triangle_mesh(goal_path, True)

    # goal_mesh = o3d.io.read_triangle_mesh(goal_path)
    # Load the goal mesh

    input_mesh = o3d.io.read_triangle_mesh(input_path, True)
    # Color the meshes
    goal_mesh.paint_uniform_color(COLORS["green"])
    input_mesh.paint_uniform_color(COLORS["blue"])

    # add shading
    goal_mesh.compute_vertex_normals()
    input_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries(
        [goal_mesh, input_mesh],
    )

    number_of_points = 50000
    goal_points = goal_mesh.sample_points_uniformly(number_of_points=number_of_points)
    input_points = input_mesh.sample_points_uniformly(number_of_points=number_of_points)

    cond_fulfilled = False
    while not cond_fulfilled:
        pts_ids_gt = pick_points(goal_points)
        pts_ids_rec = pick_points(input_points)
        cond_fulfilled = len(pts_ids_gt) == len(pts_ids_rec) and len(pts_ids_gt) >= 3
        if not cond_fulfilled:
            print("[WARN]: Did not select enough points(3), please try again")

    print(pts_ids_gt)
    print(pts_ids_rec)
    corrs = np.zeros((len(pts_ids_gt), 2))
    corrs[:, 1] = pts_ids_gt
    corrs[:, 0] = pts_ids_rec

    # corrs[:, 0] = pts_ids_gt
    # corrs[:, 1] = pts_ids_rec

    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(
        input_points, goal_points, o3d.utility.Vector2iVector(corrs)
    )
    # Show the initial alignment
    draw_registration_result(input_points, goal_points, trans_init)

    print("[INFO]: Perform point-to-point ICP refinement")
    threshold = 1  # 1cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        input_points,
        goal_points,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )
    print("[INFO]: Point-to-point ICP done")
    draw_registration_result(input_points, goal_points, reg_p2p.transformation)
    return np.array(reg_p2p.transformation).tolist(), np.array(trans_init).tolist()


def transform_mesh(mesh_path: str, transformation: list[list[float]]) -> None:
    """Transforms the mesh in place

    Args:
        mesh_path (str): Path to the mesh
        transformation (list[list[float]]): Transformation matrix
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path, True)
    mesh.transform(transformation)
    # new_name = mesh_path.split("/")[-1].split(".")[0] + "_transformed.obj"
    # print(f"[INFO]: Saving transformed mesh to {new_name}")
    # o3d.io.write_triangle_mesh(new_name, mesh)

    return mesh


if __name__ == "__main__":
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
    item = items[7]
    method = "nerfacto"

    # input_mesh_path = (
    #     "/home/testbed/Projects/mesh_manipulation/data/CADmodels/obj/" + item + "/mesh.obj"
    # )

    input_mesh_path = "data/CADmodels/obj_alligned_nerfacto/" + item + "/mesh.obj"
    # input_mesh_path = ("/home/testbed/Projects/CADmodels/obj/Axle_Front.obj")
    target_mesh_path = "data/Scenes/" + item + ".obj"
    # target_mesh_path = ("/home/testbed/Projects/mesh_manipulation/data/reconstruction/"+ method +"/meshes_cleared_scaled/"+ item + "/mesh.obj")
    transformation, init_Trans = manual_icp(target_mesh_path, input_mesh_path)
    print(transformation)

    transformation_dict_path = os.path.join("transformations_man.json")
    if not os.path.exists(transformation_dict_path):
        with open(transformation_dict_path, "w") as f:
            json.dump({}, f, indent=2)
    with open(transformation_dict_path, "r") as f:
        transformations = json.load(f)

    transformations[item + "_T_W2M"] = init_Trans
    with open(transformation_dict_path, "w") as f:
        json.dump(transformations, f, indent=2)

    transformation_dict_path = os.path.join("transformations_ICP.json")
    if not os.path.exists(transformation_dict_path):
        with open(transformation_dict_path, "w") as f:
            json.dump({}, f, indent=2)
    with open(transformation_dict_path, "r") as f:
        transformations = json.load(f)

    transformations[item + "_T_W2M"] = transformation
    with open(transformation_dict_path, "w") as f:
        json.dump(transformations, f, indent=2)

    # transformed_mesh = transform_mesh(input_mesh_path, transformation)

    # # Save mesh
    # # save_path = "/home/testbed/Projects/mesh_manipulation/data/CADmodels/obj_alligned_"+method+"/" + item
    # save_path = "/home/testbed/Projects/mesh_manipulation/data/reconstruction/bakedsdf/meshes_cleared_scaled_alligned2nerfacto/"+ item
    # os.makedirs(save_path, exist_ok=True)
    # print(f"[INFO]: Saving transformed mesh to: {save_path}")
    # file_path = os.path.join(save_path, "mesh.obj")
    # o3d.io.write_triangle_mesh(file_path, transformed_mesh)
