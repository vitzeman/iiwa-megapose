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


def nice_print(pose: list) -> None:
    x, y, z, Rz, Ry, Rx = pose
    print(
        f"x: {x:3.2f},\t y: {y:3.2f},\t z: {z:3.2f},\t Rz: {Rz:3.2f},\t Ry: {Ry:3.2f},\t Rx: {Rx:3.2f}"
    )


def align_vectors(v_i, v_t, y_rotation: float = 0.0, units="deg") -> tuple:
    """Aligns the vector v_i to v_t by rotation zyx, wheen y rotation can be given

    Args:
        v_i (_type_): _description_
        v_t (_type_): _description_
        y_rotation (float, optional): _description_. Defaults to 0.0.
        units (str, optional): Units used for the . Defaults to "deg".

    Returns:
        tuple: (Rz, Ry, Rx) euler angles in radians
    """
    if units not in ["deg", "rad"]:
        raise ValueError("units must be 'deg' or 'rad'")
    pass

    if units == "deg":
        y_rotation = np.deg2rad(y_rotation)

    vi_orig = v_i.copy()
    Rz, Ry, Rx = 0, 0, 0

    vi_xy = np.array([v_i[0], v_i[1], 0])
    vt_xy = np.array([v_t[0], v_t[1], 0])

    vi_xy = vi_xy / np.linalg.norm(vi_xy)
    vt_xy = vt_xy / np.linalg.norm(vt_xy)

    # Rotation around z axis
    Rz = np.arccos(np.dot(vi_xy, vt_xy) / (np.linalg.norm(vi_xy) * np.linalg.norm(vt_xy)))
    if vi_xy[0] * vt_xy[1] - vi_xy[1] * vt_xy[0] < 0:
        Rz = -Rz
    print("Z rotation", np.rad2deg(Rz))
    Rtx_z = rotation_matrix_z(Rz, units="rad")
    v_i = Rtx_z @ v_i.reshape(3, 1)

    print(v_i, v_t)

    # Rotation around y axis which is given by the user
    Ry = y_rotation
    Rtx_y = rotation_matrix_y(Ry, units="rad")
    v_i = Rtx_y @ v_i.reshape(3, 1)


    print(v_i, v_t)

    vi_yz = np.array([0, v_i[1,0], v_i[2,0]])
    vt_yz = np.array([0, v_t[1], v_t[2]])
    vi_yz = vi_yz / np.linalg.norm(vi_yz)
    vt_yz = vt_yz / np.linalg.norm(vt_yz)

    Rx = np.arccos(np.dot(vi_yz, vt_yz) / (np.linalg.norm(vi_yz) * np.linalg.norm(vt_yz)))
    if vi_yz[1] * vt_yz[2] - vi_yz[2] * vt_yz[1] < 0:
        Rx = -Rx

    print("Z rotation", np.rad2deg(Rx))
    Rtx_x = rotation_matrix_x(Rx, units="rad")

    print(Rtx_x)
    print(v_i)

    print(vi_orig)
    print(Rtx_z @ Rtx_y @ Rtx_x @ vi_orig.reshape(3, 1))
    v_i = Rtx_x @ v_i.reshape(3, 1)
    print(v_i, "==", v_t)
    return (Rz, Ry, Rx)



def generate_poses(center, radius):
    poses = []
    theta_gen = range(0, 21, 20)
    for theta in tqdm(theta_gen):
        theta = np.deg2rad(theta)
        for phi in range(0, 360, 60):
            # Coordinates
            phi = np.deg2rad(phi)
            x = center[0] + radius * np.sin(theta) * np.cos(phi)
            y = center[1] + radius * np.sin(theta) * np.sin(phi)
            z = center[2] + radius * np.cos(theta)

            if x > 0 or x < -500:
                continue

            if y > 0 or y < -1000:
                continue
            center = np.array(center)
            look_vector = np.array([x, y, z]) - center
            look_vector = look_vector / np.linalg.norm(look_vector)

            # Camera looks in direction of negative y axis
            look_vector = -1 * look_vector
            y_axis = np.array([0, 1, 0])
            print(type(look_vector))
            Rz, Ry, Rx = align_vectors(y_axis, look_vector)

            Rx, Ry, Rz = np.rad2deg([Rx, Ry, Rz])
            if theta == 0:
                poses.append([x, y, z, Rz, Ry, Rx])
                break

            poses.append([x, y, z, Rz, Ry, Rx])

    return poses


def visualize_pose(pose: list, ax):
    x, y, z, Rz, Ry, Rx = pose
    Rtx_x = rotation_matrix_x(Rx, units="deg")
    Rtx_y = rotation_matrix_y(Ry, units="deg")
    Rtx_z = rotation_matrix_z(Rz, units="deg")

    # Rotation is applied in the order of z, y, x
    Rtx = Rtx_z @ Rtx_y @ Rtx_x

    # Add flip by 180 degs around x axis
    # Rtx = Rtx @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # visualize the rotation pose with quiver
    vect_x = Rtx @ np.array([1, 0, 0])
    vect_y = Rtx @ np.array([0, 1, 0])
    vect_z = Rtx @ np.array([0, 0, 1])
    are_orthogonal = np.dot(vect_x, vect_y) + np.dot(vect_y, vect_z) + np.dot(vect_z, vect_x) < 1e-10
    
    print(
        f"Ortogonal? {np.dot(vect_x, vect_y) + np.dot(vect_y, vect_z) + np.dot(vect_z, vect_x) < 1e-10}"
    )
    if not are_orthogonal:
        print(np.dot(vect_x, vect_y))
        print(np.dot(vect_y, vect_z))
        print(np.dot(vect_z, vect_x))

        print(vect_x)
        print(vect_y)
        print(vect_z)

    ax.quiver(x, y, z, vect_x[0], vect_x[1], vect_x[2], length=100, color="r")
    ax.quiver(x, y, z, vect_y[0], vect_y[1], vect_y[2], length=100, color="g")
    ax.quiver(x, y, z, vect_z[0], vect_z[1], vect_z[2], length=100, color="b")
    # CAMERA LOOKS IN DIRECTION OF NEGATIVE Y AXIS
    ax.quiver(x, y, z, -vect_y[0], -vect_y[1], -vect_y[2], length=100, color="y")

    return ax


def visualize_poses_test():
    center = np.array([500, 500, 500])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot([center[0]], [center[1]], [center[2]], marker="o", color="k")

    visualize_pose([0, 0, 0, 0, 0, 0], ax)

    visualize_pose([center[0] - 250, center[1] - 250, center[2] - 250, 90, 0, 0], ax)
    visualize_pose([center[0], center[1], center[2], 90, 0, 90], ax)
    visualize_pose([center[0] - 100, center[1] - 100, center[2] - 100, 150, 0, 90], ax)
    visualize_pose([-285.51, -648.10, 469.85, 30, 0, 72], ax) # IDK WHY IT DOES NOT 

    plt.show()


def align_test():
    target = np.array([1, 1, 1])
    target =  target / np.linalg.norm(target)

    initial = np.array([0, 1, 0])

    Rz, Ry, Rx = align_vectors(initial, target, y_rotation=0, units="deg")
    print(Rz, Ry, Rx)



if __name__ == "__main__":
    # align_test()
    # visualize_poses_test()
    center = np.array([-200, -500, 0])
    radius = 500
    poses = generate_poses(center, radius)

    # Plot the xyz poses
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot([center[0]], [center[1]], [center[2]], marker="x", color="r")
    ax.quiver(0, 0, 0, 1, 0, 0, length=100, color="r", linewidth=5)
    ax.quiver(0, 0, 0, 0, 1, 0, length=100, color="g", linewidth=5)
    ax.quiver(0, 0, 0, 0, 0, 1, length=100, color="b", linewidth=5)

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
        nice_print(pose)
        ax = visualize_pose(pose, ax)

    plt.show()
