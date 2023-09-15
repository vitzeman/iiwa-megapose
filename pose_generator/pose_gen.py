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


def directed_angle(v1, v2):
    """Calculates the directed angle between two vectors

    Args:
        v1 (np.ndarray): The first vector
        v2 (np.ndarray): The second vector

    returns:
        float: The directed angle between the two vectors in radians
    """    
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    angle = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
    return angle


def align_vectors(v_i, v_t, z_rotation: float = 0.0, units="deg") -> tuple:
    """Aligns the vector v_i to v_t by rotation zyx, wheen y rotation can be given

    Args:
        v_i (_type_): _description_
        v_t (_type_): _description_
        z_rotation (float, optional): _description_. Defaults to 0.0.
        units (str, optional): Units used for the . Defaults to "deg".

    Returns:
        tuple: (Rz, Ry, Rx) euler angles in radians
    """
    # print("AT START:")
    # print("v_i", v_i)
    # print("v_t", v_t)
    if units not in ["deg", "rad"]:
        raise ValueError("units must be 'deg' or 'rad'")
    pass

    if units == "deg":
        z_rotation = np.deg2rad(z_rotation)

    vi_orig = v_i.copy()
    Rz, Ry, Rx = 0, 0, 0

    vi_xy = np.array([v_i[0], v_i[1], 0])
    vt_xy = np.array([v_t[0], v_t[1], 0])

    vi_xy_norm = np.linalg.norm(vi_xy)
    vt_xy_norm = np.linalg.norm(vt_xy)

    if vi_xy_norm == 0 or vt_xy_norm == 0:
        # print(f"vi_xy_norm: {vi_xy_norm}, vt_xy_norm: {vt_xy_norm}")
        Rz = np.deg2rad(-90)
    else:
        vi_xy = vi_xy / vi_xy_norm
        vt_xy = vt_xy / vt_xy_norm
        #Directed angle between vectors
        Rz = directed_angle(vi_xy, vt_xy)

    # print("Z rotation", np.rad2deg(Rz))

    Rtx_z = rotation_matrix_z(Rz, units="rad")
    v_i = Rtx_z @ v_i.reshape(3, 1)
    # print(v_i, v_t)
    v_i = v_i.flatten()
    # Rotation around y axis
    vi_xz = np.array([v_i[0], 0, v_i[2]])
    vt_xz = np.array([v_t[0], 0, v_t[2]])

    vi_xz_norm = np.linalg.norm(vi_xz)
    vt_xz_norm = np.linalg.norm(vt_xz)

    if vi_xz_norm == 0 or vt_xz_norm == 0:
        # print(f"vi_xz_norm: {vi_xz_norm}, vt_xz_norm: {vt_xz_norm}")
        Ry = np.deg2rad(0)
    else:
        vi_xz = vi_xz / vi_xz_norm
        vt_xz = vt_xz / vt_xz_norm

        Ry =  directed_angle(vi_xz, vt_xz)

    # print("Y rotation", np.rad2deg(Ry))
    Rtx_y = rotation_matrix_y(Ry, units="rad")
    v_i = Rtx_y @ v_i.reshape(3, 1)
    # print(v_i, v_t)
    v_i = v_i.flatten()
    # Rotation around x axis
    vi_yz = np.array([0, v_i[1], v_i[2]])
    vt_yz = np.array([0, v_t[1], v_t[2]])

    vi_yz_norm = np.linalg.norm(vi_yz)
    vt_yz_norm = np.linalg.norm(vt_yz)

    if vi_yz_norm == 0 or vt_yz_norm == 0:
        # print(f"vi_yz_norm: {vi_yz_norm}, vt_yz_norm: {vt_yz_norm}")
        Rx = np.deg2rad(0)
    else:
        vi_yz = vi_yz / vi_yz_norm
        vt_yz = vt_yz / vt_yz_norm

        Rx = directed_angle(vi_yz, vt_yz)


    # print("X rotation", np.rad2deg(Rx))
    Rtx_x = rotation_matrix_x(Rx, units="rad")
    v_i = Rtx_x @ v_i.reshape(3, 1)
    v_i = v_i.flatten()
    # Check if the vectors are aligned

    # print("v_i", v_i)
    # print("v_t", v_t)
    # print("vi_orig", vi_orig)   

    return Rz, Ry, Rx

def align_vectors2(target_vector, z_rotation, units="deg") -> tuple:
    if units not in ["deg", "rad"]:
        raise ValueError("units must be 'deg' or 'rad'")
    
    if units == "deg":
        z_rotation = np.deg2rad(z_rotation)

    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    print("AT START:")
    print("target_vector", target_vector)
    print("z_axis", z_axis)

    # Rotate the target vector around z axis
    Rz = rotation_matrix_z(z_rotation, units="rad")

    x_axis = Rz @ x_axis.reshape(3, 1)
    x_axis = x_axis.flatten()
    y_axis = Rz @ y_axis.reshape(3, 1)
    y_axis = y_axis.flatten()
    z_axis = Rz @ z_axis.reshape(3, 1)
    z_axis = z_axis.flatten()

    print(f"AFTER Z ROTATION: {np.rad2deg(z_rotation)}")
    print("target_vector", target_vector)
    print("z_axis", z_axis)

    # TODO: CHECK IF the Y_new Z_new and Target are in the same plane
    # If not rotate around y axis so that they are 
    # Y_new will be static
    # Z_new will be rotated around Y_new
    # Target will be static

    same_plane = np.dot(y_axis,  np.cross(z_axis, target_vector)) < 1e-10
    print(f"Same plane? {same_plane}")
    if not same_plane:
        # Compute the angle to align z axis to the plane given by target and y axis

        target_vector_xz = np.array([target_vector[0], 0, target_vector[2]])
        target_vector_xz = target_vector_xz / np.linalg.norm(target_vector_xz)

        z_axis_xz = np.array([z_axis[0], 0, z_axis[2]])
        z_axis_xz = z_axis_xz / np.linalg.norm(z_axis_xz)

        print("target_vector_xz", target_vector_xz)
        print("z_axis_xz", z_axis_xz)

        y_rotation = directed_angle(z_axis_xz, target_vector_xz)
        # y_rotation = directed_angle(z_axis, target_vector)
        Ry = rotation_matrix_y(y_rotation, units="rad")

        z_axis = Ry @ z_axis.reshape(3, 1)   
        z_axis = z_axis.flatten()

    else: 
        y_rotation = 0



    print(f"AFTER Y ROTATION: {np.rad2deg(y_rotation)}")
    print("target_vector", target_vector)
    print("z_axis", z_axis)

    target_vector_yz = np.array([0, target_vector[1], target_vector[2]])
    target_vector_yz = target_vector_yz / np.linalg.norm(target_vector_yz)

    z_axis_yz = np.array([0, z_axis[1], z_axis[2]])
    z_axis_yz = z_axis_yz / np.linalg.norm(z_axis_yz)

    print("target_vector_yz", target_vector_yz)
    print("z_axis_yz", z_axis_yz)

    x_rotation = directed_angle(z_axis_yz, target_vector_yz)
    # x_rotation = directed_angle(z_axis, target_vector)
    Rx = rotation_matrix_x(x_rotation, units="rad")

    z_axis = Rx @ z_axis.reshape(3, 1)

    z_axis = z_axis.flatten()

    print(f"AFTER X ROTATION: {np.rad2deg(x_rotation)}")
    print("target_vector", target_vector)
    print("z_axis", z_axis)

    print("")
    return z_rotation, y_rotation, x_rotation




def generate_poses(center, radius):
    poses = []
    theta_gen = range(0, 21, 20)
    should_break = False
    for theta in tqdm(theta_gen):
        theta = np.deg2rad(theta)
        if should_break:
            break
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
            look_vector = center - np.array([x, y, z])
            look_vector = look_vector / np.linalg.norm(look_vector)

            z_axis = np.array([0, 0, 1])
            # print(type(look_vector))
            # Rz, Ry, Rx = align_vectors(z_axis, look_vector)
            Rz, Ry, Rx = align_vectors2(look_vector, -90, units="deg")
            Rz, Ry, Rx = np.rad2deg([Rz, Ry, Rx])

            # Rz, Ry, Rx = angle_to_range_180(Rz, Ry, Rx)

            if theta == 0:
                poses.append([x, y, z, Rz, Ry, Rx])
                should_break = False
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
    vect_x = Rtx @ np.array([1, 0, 0]).reshape(3, 1)
    vect_y = Rtx @ np.array([0, 1, 0]).reshape(3, 1)
    vect_z = Rtx @ np.array([0, 0, 1]).reshape(3, 1)
    
    vect_x = vect_x.flatten()
    vect_y = vect_y.flatten()
    vect_z = vect_z.flatten()

    are_orthogonal = np.dot(vect_x, vect_y) + np.dot(vect_y, vect_z) + np.dot(vect_z, vect_x) < 1e-10
    print(
        f"Ortogonal? {are_orthogonal}"
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
    # vect_1 = np.array([1, 0, 0])
    # vect_2 = np.array([-1, 1, 0])

    # angle = directed_angle(vect_1, vect_2)
    # print(np.rad2deg(angle))
    # Rtx_z = rotation_matrix_z(angle, units="rad")


    # Rtx = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # print(Rtx)
    # # vect_1 = Rtx @ vect_1.reshape(2, 1)
    # vect_1 = Rtx_z @ vect_1.reshape(3, 1)
    # print(vect_1.flatten())
    # print(vect_2)

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
        nice_pose_print(pose)
        ax = visualize_pose(pose, ax)

    plt.show()
