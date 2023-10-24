import cv2
import numpy as np

SIZES = {
    "A4": (210, 297),
    "A3": (297, 420),
    "A2": (420, 594),
    "A1": (594, 841),
    "A0": (841, 1189),
} # in mm

DPI = 300
TAG_SIZE_MM = 30 # in mm
TAG_SIZE_PX = int(TAG_SIZE_MM * DPI / 25.4) # in px


def get_pattern_arruco():
    """
    Returns a pattern of the ArUco markers.
    """
    # p1 = np.array([[0, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0], [0.1, 0, 0]], dtype=np.float32)
    # p2 = np.array([[2, 0, 0], [2, 1, 0], [3, 1, 0], [3, 0, 0]], dtype=np.float32)
    # objPoints = np.array([p1, p2])

    objPoints = []
    for i in range(2):
        for j in range(2):
            objPoints.append(np.array([[i, j, 0], [i, j+1, 0], [i+1, j+1, 0], [i+1, j, 0]], dtype=np.float32))

    objPoints = np.array(objPoints, dtype=np.float32)
    print(objPoints.shape)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    ids = np.array([0, 1, 2,3])
    board = cv2.aruco.Board(objPoints, dictionary, ids)
    return board

if __name__ == "__main__":
    board = get_pattern_arruco()
    print(board)
    img = board.generateImage((500, 500), marginSize=10, borderBits=1)
    print(img.shape)
    cv2.imshow("board", img) 

    cv2.waitKey(0)
    cv2.destroyAllWindows()
