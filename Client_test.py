from mlsocket import MLSocket
import cv2
import time

import numpy as np

HOST = "10.35.129.250"
PORT = 65432

# Make an ndarray
data = np.array([1, 2, 3, 4])

# Make a keras model


def fce(socket, img: np.ndarray, bbox: np.ndarray, id: np.ndarray):
    socket.send(img)  # Send the data
    socket.send(bbox)
    socket.send(id)

    pose = socket.recv(1024)
    return pose


# cv2.namedWindow("TestClient", cv2.WINDOW_NORMAL)
# Send data
with MLSocket() as s:
    s.connect((HOST, PORT))  # Connect to the port and host
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    print(img.shape)
    print(type(img))
    i = 0
    while True:
        # s.send(img)  # Send the data

        # bbox = np.array([1, 2, 3, 4])
        # s.send(bbox)

        # id = np.array((1))
        # s.send(id)

        # # Receive the data
        # data = s.recv(1024)q

        # print(f"Data received: {data.shape}")

        pose = fce(s, img, np.array([1, 2, 3, 4]), np.array([i]))

        print(f"{i}\tData received: {pose}")
        i += 1
        if i == 11:
            break
        # time.sleep(2)