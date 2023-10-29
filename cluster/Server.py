from mlsocket import MLSocket
import numpy as np
import time
import cv2

HOST = "127.0.0.1"
PORT = 65432

# cv2.namedWindow("TestServer", cv2.WINDOW_NORMAL)


def process_iteration(socket):
    # Receive the data
    data = socket.recv(1024)
    bbox = socket.recv(1024)
    id = socket.recv(1024)

    print(f"Data received: {data.shape}")
    print(f"Data received: {bbox}")
    print(f"Data received: {id}")

    # Sent back the data
    # TODO: Add function to process the data and get the pose(quaternion + translation)
    pose = np.array([1, 2, 3, 4, 5, 6, 7])  # quaternion + translation
    socket.send(pose)
    return


with MLSocket() as s:
    s.bind((HOST, PORT))

    s.listen()
    conn, address = s.accept()

    with conn:
        while True:
            # data = conn.recv(1024)

            # print(f"Data received: {data.shape}")
            # cv2.imshow("TestServer", data)

            # # recieve another bbox
            # bbox = conn.recv(1024)
            # print(f"Data received: {bbox}")

            # # Sent back the data
            # data[:, :, :2] = 0
            # conn.send(data[::-1])

            process_iteration(conn)
