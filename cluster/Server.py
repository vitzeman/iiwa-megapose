from mlsocket import MLSocket
import numpy as np
import time
import cv2

import argparse

# HOST = "192.168.1.156" # My home IP
HOST = "127.0.0.1"
PORT = 65432

# cv2.namedWindow("TestServer", cv2.WINDOW_NORMAL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)

    return parser.parse_args()


def process_iteration(socket):
    """Proccess one iteration of the from the client provided data and send back result

    Args:
        socket (_type_): _description_
    """
    # Receive the data
    data = socket.recv(1024)
    bbox = socket.recv(1024)
    id = socket.recv(1024)

    print(f"Data received: {data.shape}")
    print(f"Data received: {bbox}")
    print(f"Data received: {id}")

    # Sent back the data
    # TODO: Add function to process the data and get the pose(quaternion + translation)
    # TODO: Bassically write in megapose function to get the pose
    pose = np.array([1, 2, 3, 4, 5, 6, 7])  # quaternion + translation
    socket.send(pose)
    return


# with MLSocket() as s:
#     s.bind((HOST, PORT))

#     s.listen()
#     conn, address = s.accept()

#     with conn:
#         while True:
#             # data = conn.recv(1024)

#             # print(f"Data received: {data.shape}")
#             # cv2.imshow("TestServer", data)

#             # # recieve another bbox
#             # bbox = conn.recv(1024)
#             # print(f"Data received: {bbox}")

#             # # Sent back the data
#             # data[:, :, :2] = 0
#             # conn.send(data[::-1])

#             process_iteration(conn)

if __name__ == "__main__":
    args = parse_args()

    host = args.host
    port = args.port

    with MLSocket() as s:
        s.connect((HOST, PORT))

        s.listen()
        conn, address = s.accept()
        with conn:
            while True:
                process_iteration(conn)
