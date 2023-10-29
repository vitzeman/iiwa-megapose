import argparse
import os
import time

from mlsocket import MLSocket
import cv2
import paramiko
import numpy as np

HOST = "127.0.0.1"
PORT = 65432

# Make an ndarray
data = np.array([1, 2, 3, 4])


# Make a keras model
def parse_args():
    # TODO: Not tested
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)
    parser.add_argument("--user", type=str, default="zemanvit")
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument("--init_server", type=bool, default=False)

    return parser.parse_args()


def start_server(host, port, usr, pwd):
    # TODO: Not tested
    # Start the server
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=usr, password=pwd)
    ssh.exec_command("conda activate megapose")
    ssh.exec_command(f"python Project/megapose/Server.py --host {host} --port {port}")
    time.sleep(1)
    return ssh


def stop_server(ssh):
    # TODO: Not tested
    ssh.send(chr(3))
    ssh.exec_command("conda deactivate")
    ssh.close()


def proccess_input_data(
    socket, img: np.ndarray, bbox: np.ndarray, id: np.ndarray
) -> np.ndarray:
    """Sents the data to the server and waits for the response

    Args:
        socket (): Socket to send the data
        img (np.ndarray): Image to send, shape (H, W, C)
        bbox (np.ndarray): Bounding box to send shape, (4,)
        id (np.ndarray): Id of the object to send shape, (1,)

    Returns:
        np.ndarray: Pose of the object, shape (7,)
    """
    # TODO: Add check for the shape of the data

    socket.send(img)
    socket.send(bbox)
    socket.send(id)

    pose = socket.recv(1024)
    return pose


# cv2.namedWindow("TestClient", cv2.WINDOW_NORMAL)
# Send data
# with MLSocket() as s:
#     s.connect((HOST, PORT))  # Connect to the port and host
#     img = cv2.imread("aruco_pattern/board.png")
#     print(img.shape)
#     print(type(img))
#     while True:
#         # s.send(img)  # Send the data

#         # bbox = np.array([1, 2, 3, 4])
#         # s.send(bbox)

#         # id = np.array((1))
#         # s.send(id)

#         # # Receive the data
#         # data = s.recv(1024)q

#         # print(f"Data received: {data.shape}")

#         pose = fce(s, img, np.array([1, 2, 3, 4]), np.array([1]))

#         print(f"Data received: {pose}")

if __name__ == "__main__":
    args = parse_args()
    host = args.host
    port = args.port
    usr = args.user
    pwd = args.password

    init_server = args.init_server
    if init_server:
        # Start the server at the given host
        start_server(host, usr, pwd)

    with MLSocket() as s:
        s.connect((HOST, PORT))  # Connect to the port and host
        img = cv2.imread("aruco_pattern/board.png")
        print(img.shape)
        print(type(img))
        while True:
            pose = proccess_input_data(s, img, np.array([1, 2, 3, 4]), np.array([1]))

            print(f"Data received: {pose}")

    # Stop the server
    if init_server:
        stop_server(ssh)
