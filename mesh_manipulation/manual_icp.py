import open3d as o3d
import numpy as np


# Constants
ROOT_PATH = "~/Documents/megapose"
cropped_meshes_name = "extended_non_transformed_scaled_clean_meshes_nerf"

# For some reason the meshes cannot start wuth a ~/ so use relative path
nerfactor_mesh_path = "../megapose/extended_non_transformed_scaled_clean_meshes_nerf/nerfacto_all/e08_3d2/mesh.obj"
gt_mehs_path = "../megapose/CADmodels/obj/Chassis.obj"


def pick_points(pcd):
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


if __name__ == "__main__":
    rec_mesh = o3d.io.read_triangle_mesh(nerfactor_mesh_path)
    gt_mesh = o3d.io.read_triangle_mesh(gt_mehs_path)
    # o3d.visualization.draw_geometries([rec_mesh, gt_mesh])
    # Color the meshes
    rec_mesh.paint_uniform_color([0.1, 0.1, 0.7])
    gt_mesh.paint_uniform_color([0.1, 0.7, 0.1])
    # Add shading
    rec_mesh.compute_vertex_normals()
    gt_mesh.compute_vertex_normals()

    gt_points = gt_mesh.sample_points_uniformly(number_of_points=50000)
    rec_points = rec_mesh.sample_points_uniformly(number_of_points=50000)
    # Draw
    o3d.visualization.draw_geometries(
        [rec_mesh, gt_mesh],
    )
    pts_ids_gt = pick_points(gt_points)
    pts_ids_rec = pick_points(rec_points)
    print(pts_ids_gt)
    print(pts_ids_rec)
    corrs = np.zeros((len(pts_ids_gt), 2))
    corrs[:, 0] = pts_ids_gt
    corrs[:, 1] = pts_ids_rec

    # Manualy select corresponding  points and then align them with ICP
    # picked_id_source = pick_points(rec_mesh)
