from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt


def generate_poses(center, radius):
    poses = []
    theta_gen = range(0, 21, 20)
    for theta in tqdm(theta_gen):
        theta = np.deg2rad(theta)
        for phi in range(0, 360, 60):
            phi = np.deg2rad(phi)
            x = center[0] + radius * np.sin(theta) * np.cos(phi)
            y = center[1] + radius * np.sin(theta) * np.sin(phi)
            z = center[2] + radius * np.cos(theta)

            if x > 0 or x < -500:
                continue

            if y>0 or y < -1000:
                continue
            look_vector = np.array([x, y, z]) - center
            look_vector = look_vector / np.linalg.norm(look_vector)

            if theta == 0:
                poses.append([x, y, z, 0, 0, 0])
                break

            poses.append([x, y, z, theta, phi, 0])    

    return poses



if __name__ == '__main__':
    center = np.array([-200, -500, 0])
    radius = 500
    poses = generate_poses(center, radius)

    # Plot the xyz poses
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([center[0]], [center[1]], [center[2]], marker='o')
    for pose in poses:
        print(pose[2])
        ax.scatter(pose[0], pose[1], pose[2], marker='o')
        vector = center - np.array([pose[0], pose[1], pose[2]]) 
        vector = vector / np.linalg.norm(vector)
        ax.quiver(pose[0], pose[1], pose[2], vector[0], vector[1], vector[2], length=100)

    plt.show()