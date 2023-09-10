import os
import open3d as o3d
import json
import numpy as np


def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def visualize_cameras(colored_camera_dicts, sphere_radius, camera_size=0.1, geometry_file=None, geometry_type='mesh'):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]

    idx = 0
    for color, camera_dict in colored_camera_dicts:
        idx += 1

        cnt = 0
        frustums = []
        for img_name in sorted(camera_dict.keys()):
            K = np.array(camera_dict[img_name]['K']) #.reshape((4, 4))
            W2C = np.array(camera_dict[img_name]['W2C']) #.reshape((4, 4))
            C2W = np.linalg.inv(W2C)
            img_size = camera_dict[img_name]['img_size']
            frustums.append(get_camera_frustum(img_size, K, W2C, frustum_length=camera_size, color=color))
            cnt += 1
        cameras = frustums2lineset(frustums)
        things_to_draw.append(cameras)

    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)

    o3d.visualization.draw_geometries(things_to_draw)


from rotation_helper import rotation_matrix
out_dict = {}
head, tail = os.path.split(os.path.split(os.getcwd())[0])
pose_dir = os.path.join(head, tail, 'dataset', 'aruco_1', 'pose')
json_file = open(os.path.join(pose_dir, 'raw.json'))

head, tail = os.path.split(os.path.split(os.getcwd())[0])
config_dir = os.path.join(head, tail, 'config')
camera_file = open(os.path.join(config_dir, 'camera.json'))
camera = json.load(camera_file)
camera_matrix = np.array(camera['camera_matrix'])
dist_coeff = np.array(camera['dist_coeff'])

raw_data = json.load(json_file)

i = 0
for key in raw_data:
    T = np.zeros((4, 4))
    T[0, 3] = raw_data[key]['RX'] * 0.001
    T[1, 3] = raw_data[key]['RY'] * 0.001
    T[2, 3] = raw_data[key]['RZ'] * 0.001
    rot = rotation_matrix( raw_data[key]['RA'],  raw_data[key]['RB'], raw_data[key]['RC'], order= 'zyx')
    T[:3, :3] = rot
    T[3, 3] = 1

    T = np.linalg.inv(T)
    # T[:3, :3] = rot

    out_dict[str(i) + '.jpg'] = {}
    out_dict[str(i) + '.jpg']['K'] = camera_matrix.tolist()
    out_dict[str(i) + '.jpg']['img_size'] = [2448, 2048]
    out_dict[str(i) + '.jpg']['W2C'] = T.tolist()
    i +=1


with open(os.path.join(pose_dir, "pose1.json"), "w") as outfile:
    json.dump(out_dict, outfile)

sphere_radius = 1.
# train_cam_dict = json.load(open(os.path.join(base_dir, 'train/cam_dict_norm.json')))
# test_cam_dict = json.load(open(os.path.join(base_dir, 'test/cam_dict_norm.json')))
# path_cam_dict = json.load(open(os.path.join(base_dir, 'camera_path/cam_dict_norm.json')))
train_cam_dict = json.load(open(os.path.join(pose_dir,  'pose1.json')))  #Green
# test_cam_dict = json.load(open(os.path.join(pose_dir,  'pose.json')))  #Blue
path_cam_dict = json.load(open(os.path.join(pose_dir, 'marker_pose.json')))

camera_size = 0.05
colored_camera_dicts = [
                        ([0, 1, 0], train_cam_dict),
                        # ([0, 0, 1], test_cam_dict),
                        ([1, 0, 0], path_cam_dict)
                        ]

geometry_file = os.path.join(pose_dir, 'mesh', '.ply')
geometry_type = 'mesh'

visualize_cameras(colored_camera_dicts, sphere_radius,
                  camera_size=camera_size, geometry_file=geometry_file, geometry_type=geometry_type)
