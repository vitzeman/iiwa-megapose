import json
import os

from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

# from scipy.spatial.transform import Rotation as R

# from scipy.spatial.transform import Rotation as R.


def rotation_angles(matrix, order):
    """
    input
        matrix = 3x3 rotation matrix (numpy array)
        oreder(str) = rotation order of x, y, z : e.g, rotation XZY -- 'xzy'
    output
        theta1, theta2, theta3 = rotation angles in rotation order
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    if order == "xzx":
        theta1 = np.arctan(r31 / r21)
        theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(-r13 / r12)

    elif order == "xyx":
        theta1 = np.arctan(-r21 / r31)
        theta2 = np.arctan(-r31 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(r12 / r13)

    elif order == "yxy":
        theta1 = np.arctan(r12 / r32)
        theta2 = np.arctan(r32 / (r22 * np.cos(theta1)))
        theta3 = np.arctan(-r21 / r23)

    elif order == "yzy":
        theta1 = np.arctan(-r32 / r12)
        theta2 = np.arctan(-r12 / (r22 * np.cos(theta1)))
        theta3 = np.arctan(r23 / r21)

    elif order == "zyz":
        theta1 = np.arctan(r23 / r13)
        theta2 = np.arctan(r13 / (r33 * np.cos(theta1)))
        theta3 = np.arctan(-r32 / r31)

    elif order == "zxz":
        theta1 = np.arctan(-r13 / r23)
        theta2 = np.arctan(-r23 / (r33 * np.cos(theta1)))
        theta3 = np.arctan(r31 / r32)

    elif order == "xzy":
        theta1 = np.arctan(r32 / r22)
        theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
        theta3 = np.arctan(r13 / r11)

    elif order == "xyz":
        theta1 = np.arctan(-r23 / r33)
        theta2 = np.arctan(r13 * np.cos(theta1) / r33)
        theta3 = np.arctan(-r12 / r11)

    elif order == "yxz":
        theta1 = np.arctan(r13 / r33)
        theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
        theta3 = np.arctan(r21 / r22)

    elif order == "yzx":
        theta1 = np.arctan(-r31 / r11)
        theta2 = np.arctan(r21 * np.cos(theta1) / r11)
        theta3 = np.arctan(-r23 / r22)

    elif order == "zyx":
        theta1 = np.arctan(r21 / r11)
        theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
        theta3 = np.arctan(r32 / r33)

    elif order == "zxy":
        theta1 = np.arctan(-r12 / r22)
        theta2 = np.arctan(r32 * np.cos(theta1) / r22)
        theta3 = np.arctan(-r31 / r33)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return (theta1, theta2, theta3)


def rotation_matrix_x(theta, units="deg"):
    if units == "deg":
        theta = np.deg2rad(theta)
    if units not in ["deg", "rad"]:
        raise ValueError("units must be 'deg' or 'rad'")
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def rotation_matrix_y(theta, units="deg"):
    if units == "deg":
        theta = np.deg2rad(theta)
    if units not in ["deg", "rad"]:
        raise ValueError("units must be 'deg' or 'rad'")
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def rotation_matrix_z(theta, units="deg"):
    if units == "deg":
        theta = np.deg2rad(theta)
    if units not in ["deg", "rad"]:
        raise ValueError("units must be 'deg' or 'rad'")
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def rotation_matrix_zyx(Rz, Ry, Rx, units="deg"):
    if units == "deg":
        Rz, Ry, Rx = np.deg2rad([Rz, Ry, Rx])

    if units not in ["deg", "rad"]:
        raise ValueError("units must be 'deg' or 'rad'")

    pass


def nice_pose_print(pose: list) -> None:
    x, y, z, Rz, Ry, Rx = pose
    print(
        f"x: {x:3.2f},\t y: {y:3.2f},\t z: {z:3.2f},\t Rz: {Rz:3.2f},\t Ry: {Ry:3.2f},\t Rx: {Rx:3.2f}"
    )


def angle_to_range_180(*angles: float) -> float:
    """Converts the given angles to the range [-180, 180]

    Args:
        angles (float): The angle to convert

    Returns:
        float: The converted angle
    """
    converted = []
    for angle in angles:
        if angle > 180:
            angle = angle - 360

        if angle < -180:
            angle = angle + 360

        converted.append(angle)

    return converted


def directed_angle(source_vector: np.ndarray, target_vector: np.ndarray, axis) -> float:
    """Calculates the angle from source_vector to target_vector around axis
    ONLY WORKS FOR VECTORS WHICH ARE ALREADY IN THE PLANE DEFINED BY AXIS

    Args:
        source_vector (np.ndarray): _description_
        target_vector (np.ndarray): _description_
        axis (np.ndarray): Axis to rotate around

    Returns:
        float: np.ndarray
    """
    # Check if plane given by source and target is the same as the plane defined by axis
    # if np.dot(source_vector, axis) > 1e-10 or np.dot(target_vector, axis) > 1e-10:
    #     raise ValueError(
    #         "source and target vectors must be in the plane defined by axis"
    #     )

    source_vector = source_vector / np.linalg.norm(source_vector)
    target_vector = target_vector / np.linalg.norm(target_vector)

    angle = np.arctan2(
        np.linalg.norm(np.cross(source_vector, target_vector)),
        np.dot(source_vector, target_vector),
    )

    # Check if the angle should be positive or negative
    triple_dot_product = np.dot(source_vector, np.cross(target_vector, axis))
    if triple_dot_product < 0:
        angle = -angle
    # print(angle)
    return angle


def vectors_in_same_plane(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, tol: float = 1e-14
):
    """Checks if the given vectors are in the same plane

    Args:
        v1  np.ndarray): First vector
        v2 (np.ndarray): Second vector
        v3 (np.ndarray): Third vector
        to
    """
    ret = np.abs(np.dot(v1, np.cross(v2, v3))) < 1e-14
    # if not ret:
        # print(np.dot(v1, np.cross(v2, v3)))

    return ret


def orthogonal_projection(vector: np.ndarray, basis: list) -> np.ndarray:
    """Projects the given vector onto the plane defined by the given basis

    Args:
        vector (np.ndarray): Vector to project
        basis (list): List of vectors defining the plane to project onto

    Returns:
        np.ndarray: The projected vector
    """
    if len(basis) != 2:
        raise ValueError("basis must be a list of 2 vectors")
    if len(vector) != 3:
        raise ValueError("vector must be a 3D vector")

    if np.dot(basis[0], basis[1]) > 1e-10:
        raise ValueError("basis vectors must be orthogonal")

    b1 = basis[0]
    b2 = basis[1]

    b1 = b1 / np.linalg.norm(b1)
    b2 = b2 / np.linalg.norm(b2)

    projected_vector = (
        np.dot(vector, b1) / np.dot(b1, b1) * b1
        + np.dot(vector, b2) / np.dot(b2, b2) * b2
    )

    check_same_plane = np.dot(projected_vector, np.cross(b1, b2)) < 1e-10
    if not check_same_plane:
        raise ValueError("The projected vector is not in the same plane as the basis")
    return projected_vector


def align_vectors(
    target_vector, z_rotation, units="deg", ax=None, location=None
) -> tuple:
    if units not in ["deg", "rad"]:
        raise ValueError("units must be 'deg' or 'rad'")

    if units == "deg":
        z_rotation = np.deg2rad(z_rotation)
    # print("target_vector", target_vector)

    # Base vectors
    base_x = np.array([1, 0, 0])
    base_y = np.array([0, 1, 0])
    base_z = np.array([0, 0, 1])

    # Z rotation is given by the user
    Rtx_z = rotation_matrix_z(z_rotation, units="rad")

    # Rotate the base vectors
    base_x = Rtx_z @ base_x.reshape(3, 1)
    base_x = base_x.flatten()
    base_y = Rtx_z @ base_y.reshape(3, 1)
    base_y = base_y.flatten()
    base_z = Rtx_z @ base_z.reshape(3, 1)
    base_z = base_z.flatten()

    if vectors_in_same_plane(base_y, base_z, target_vector):
        # print("Vectors in same plane")
        # print("base_y", base_y)
        # print("base_z", base_z)
        # print("target_vector", target_vector)
        # print(np.dot(base_y, np.cross(base_z, target_vector)))
        y_rotation = 0
    else:
        target_projection_bxbz = orthogonal_projection(target_vector, [base_x, base_z])
        y_rotation = directed_angle(base_z, target_projection_bxbz, base_y)

    # print(y_rotation)
    Rtx_y = rotation_matrix_y(y_rotation, units="rad")

    base_x = np.array([1, 0, 0])
    base_y = np.array([0, 1, 0])
    base_z = np.array([0, 0, 1])
    # Rotate the base vectors
    base_x = Rtx_z @ Rtx_y @ base_x.reshape(3, 1)
    base_x = base_x.flatten()
    base_y = Rtx_z @ Rtx_y @ base_y.reshape(3, 1)
    base_y = base_y.flatten()
    base_z = Rtx_z @ Rtx_y @ base_z.reshape(3, 1)
    base_z = base_z.flatten()

    target_projection_bybz = orthogonal_projection(target_vector, [base_y, base_z])
    x_rotation = directed_angle(base_z, target_projection_bybz, base_x)

    return z_rotation, y_rotation, x_rotation


def generate_poses(
    center: np.ndarray,
    radius: float,
    Rz:float = -90,
    theta_gen: range = range(0, 61, 20),
    phi_gen: range = range(0, 360, 60),
    x_limits: list = None,
    y_limits: list = None,
    z_limits: list = None,

):
    """Generates poses on a sphere around the given center with the given radius


    Args:
        center (np.ndarray): Center of the spehere where camera should look at
        radius (float): Radius of the sphere
        Rz (float, optional): Rotation around z axis given by user in degreese. Defaults to -90.
        theta_gen (range, optional): Angles to generate poses. The angles are defined as angle from z axis. Defaults to range(0,90,20).
        phi_gen (range, optional): Angles to generate poses. The angles are defiend as angle from x axis. Defaults to (0, 360,60).
        x_limits (list, optional): Limits for x axis. Defaults to None.
        y_limits (list, optional): Limits for y axis. Defaults to None.
        z_limits (list, optional): Limits for z axis. Defaults to None.
    Returns:
        _type_: _description_
    """
    # Maybe add warning if the given limits are not in correct order
    if x_limits is not None:
        x_min, x_max = min(x_limits), max(x_limits)

    if y_limits is not None:
        y_min, y_max = min(y_limits), max(y_limits)

    if z_limits is not None:
        z_min, z_max = min(z_limits), max(z_limits)

    poses = []
    should_break = False
    i = 1
    for theta in tqdm(theta_gen):
        theta = np.deg2rad(theta)
        if should_break:
            break
        for phi in phi_gen:
            # Coordinates
            phi = np.deg2rad(phi)
            x = center[0] + radius * np.sin(theta) * np.cos(phi)
            y = center[1] + radius * np.sin(theta) * np.sin(phi)
            z = center[2] + radius * np.cos(theta)

            if x_limits is not None and (x < x_min or x > x_max):
                continue

            if y_limits is not None and (y < y_min or y > y_max):
                continue

            if z_limits is not None and (z < z_min or z > z_max):
                continue
            
            i += 1
            center = np.array(center)
            look_vector = center - np.array([x, y, z])
            look_vector = look_vector / np.linalg.norm(look_vector)

            z_axis = np.array([0, 0, 1])
            Rz, Ry, Rx = align_vectors(look_vector, z_rotation=Rz, location=(x, y, z))

            Rz, Ry, Rx = np.rad2deg([Rz, Ry, Rx])

            Rz, Ry, Rx = angle_to_range_180(Rz, Ry, Rx)

            if (
                theta == 0
            ):  # Only add one pose for theta = 0, that is looking straight down
                poses.append([x, y, z, Rz, Ry, Rx])
                should_break = False
                break

            poses.append([x, y, z, Rz, Ry, Rx])

    return poses


def save_poses_to_json(poses: list, filename: str):
    """Saves the given poses to a json file

    Args:
        poses (list): List of poses to save
        filename (str): Filename to save to
    """    
    dictionary = {}
    for e,pose in enumerate(poses):
        X, Y, Z, RA, RB, RC = pose
        pose = {
            "X": X,
            "Y": Y,
            "Z": Z,
            "RA": RA,
            "RB": RB,
            "RC": RC,
        }
        dictionary[str(e)] = pose

    with open(filename, "w") as f:
        json.dump(dictionary, f, indent=2)

    return None

    
def visualize_pose(pose: list, ax):
    # nice_pose_print(pose)
    x, y, z, Rz, Ry, Rx = pose
    Rtx_x = rotation_matrix_x(Rx, units="deg")
    Rtx_y = rotation_matrix_y(Ry, units="deg")
    Rtx_z = rotation_matrix_z(Rz, units="deg")

    # Rotation is applied in the order of z, y, x
    Rtx = Rtx_z @ Rtx_y @ Rtx_x

    # Add flip by 180 degs around x axis
    # Rtx = Rtx @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # visualize the rotation pose with quiver
    vect_x = Rtx @ np.array([1, 0, 0]).reshape(3, 1)
    vect_y = Rtx @ np.array([0, 1, 0]).reshape(3, 1)
    vect_z = Rtx @ np.array([0, 0, 1]).reshape(3, 1)

    vect_x = vect_x.flatten()
    vect_y = vect_y.flatten()
    vect_z = vect_z.flatten()

    are_orthogonal = (
        np.dot(vect_x, vect_y) + np.dot(vect_y, vect_z) + np.dot(vect_z, vect_x) < 1e-10
    )
    # print(f"Ortogonal? {are_orthogonal}")

    if not are_orthogonal:
        print("Transformed base vectors are not orthogonal")
        print(np.dot(vect_x, vect_y))
        print(np.dot(vect_y, vect_z))
        print(np.dot(vect_z, vect_x))

        print(vect_x)
        print(vect_y)
        print(vect_z)

    ax.quiver(x, y, z, vect_x[0], vect_x[1], vect_x[2], length=100, color="r")
    ax.quiver(x, y, z, vect_y[0], vect_y[1], vect_y[2], length=100, color="g")
    ax.quiver(x, y, z, vect_z[0], vect_z[1], vect_z[2], length=100, color="b")
    return ax

if __name__ == "__main__":
    center = np.array([-200, -500, 0])
    radius = 500

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot([center[0]], [center[1]], [center[2]], marker="x", color="r")
    ax.quiver(0, 0, 0, 1, 0, 0, length=100, color="r", linewidth=5)
    ax.quiver(0, 0, 0, 0, 1, 0, length=100, color="g", linewidth=5)
    ax.quiver(0, 0, 0, 0, 0, 1, length=100, color="b", linewidth=5)

    poses = generate_poses(center, radius)
    for pose in poses:
        ax.scatter(pose[0], pose[1], pose[2], marker="o")
        vector = center - np.array([pose[0], pose[1], pose[2]])
        vector = vector / np.linalg.norm(vector)
        ax.quiver(
            pose[0],
            pose[1],
            pose[2],
            vector[0],
            vector[1],
            vector[2],
            length=100,
            color="k",
        )
        ax = visualize_pose(pose, ax)

    plt.show()
