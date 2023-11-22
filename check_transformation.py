import numpy as np
from scipy.spatial.transform import Rotation as R


np.set_printoptions(suppress=True)

# current postion of robot in world frame
T_f2W = np.eye(4)
T_f2W[:3, 3] = np.array([-200, -500, 500])
T_f2W[:3, :3] = R.from_euler('ZYX', [90, 0, 180], degrees=True).as_matrix()


## apply flange transformation

print("flange transformation in world frame")
print(T_f2W)

print(" world transformation in flange frame")
print(np.linalg.inv(T_f2W))


# camera extrinsics
camera_transformation = np.eye(4)
camera_translation = np.array([3, -90, 60])
camera_rotation = np.array([0, 0, 0])
camera_transformation[:3, :3] = R.from_euler('zyx', camera_rotation, degrees=True).as_matrix()
camera_transformation[:3, 3] = camera_translation
print("camera transformation respect to flange")
print(camera_transformation)


# camera postion in world frame
camera_position = np.matmul(T_f2W, camera_transformation)
print("camera postion in world space")
print(camera_position)
camera_position = np.linalg.inv(camera_position)
print("world postion in camera frame")
print(camera_position)


# Example of object frame in the camera frame
object_tranlation = np.array([0,-10, 350])
object_rotation = np.array([0, 0,-90])
object_matrix = np.eye(4)
object_matrix[:3, 3] = object_tranlation
object_matrix[:3, :3] = R.from_euler('zyx', object_rotation, degrees=True).as_matrix()

print("object in camera frame")
print(object_matrix)

# object frame in world frame
object_world = np.matmul(np.linalg.inv(camera_position), object_matrix)
print("object in world frame")
print(object_world)

print( " object rotation in world frame")
print(R.from_matrix(object_world[:3, :3]).as_euler('zyx', degrees=True))

