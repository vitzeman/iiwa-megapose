from typing import Union, Tuple
import json

import cv2
import numpy as np

SIZES = {
    "A4": (210, 297),
    "A3": (297, 420),
    "A2": (420, 594),
    "A1": (594, 841),
    "A0": (841, 1189),
}  # in mm


DPI = 300
TAG_SIZE_MM = 30  # in mm
TAG_SIZE_PX = int(TAG_SIZE_MM * DPI / 25.4)  # in px


def get_rectangle_pattern(
    tag_size: int,
    dpi: int,
    paper_size: Union[str, Tuple[int, int]] = "A4",
    margins: int = 10,
) -> cv2.aruco.Board:
    """Creates aruco pattern with tags at the edges of the rectangle.

    Args:
        tag_size (int): Size of the tag in mm.
        dpi (int): DPI of the image.
        paper_size (Union[str, Tuple[int, int]], optional): Size of the paper in milimeters or A# format. Defaults to "A4".
        margins (int, optional): Margins in mm. Defaults to 10.

    Returns:
        cv2.aruco.Board: Aruco board with the pattern.
    """
    if isinstance(paper_size, str):
        paper_size = SIZES[paper_size]
    elif isinstance(paper_size, tuple):
        paper_size = paper_size
        assert len(paper_size) == 2
    else:
        raise ValueError("paper_size must be either str or tuple")

    tag_size_px = int(tag_size * dpi / 25.4)
    margins_px = int(margins * dpi / 25.4)
    tag_size_mm = tag_size
    print(f"Tag size in px: {tag_size_px}")

    paper_size_px = (int(paper_size[0] * dpi / 25.4), int(paper_size[1] * dpi / 25.4))

    paper_x_px = paper_size_px[0] - 2 * margins_px
    paper_y_px = paper_size_px[1] - 2 * margins_px
    print(f"Paper size in px: {paper_size_px}")
    print(f"Paper size in mm: {paper_size}")

    # number of tags in x and y direction
    num_x = paper_x_px // tag_size_px
    num_y = paper_y_px // tag_size_px

    # gap between tags in x and y direction
    gap_x = paper_x_px % tag_size_px / (num_x - 1)
    gap_y = paper_y_px % tag_size_px / (num_y - 1)

    print(f"Gap in x direction: {gap_x}")
    print(f"Gap in y direction: {gap_y}")

    while gap_x < 1 / 3 * tag_size_px:
        num_x -= 1
        gap_x = (paper_x_px - num_x * tag_size_px) / (num_x - 1)

    while gap_y < 1 / 3 * tag_size_px:
        num_y -= 1
        gap_y = (paper_y_px - num_y * tag_size_px) / (num_y - 1)

    print(f"Number of tags in x direction: {num_x}")
    print(f"Number of tags in y direction: {num_y}")
    print(f"Gap in x direction: {gap_x}")
    print(f"Gap in y direction: {gap_y}")

    obj_points = []
    ids = []
    idx = 0
    output_dict = {
        "paper_size_px": paper_size_px,
        "paper_size_mm": paper_size,
        "tag_size_px": tag_size_px,
        "tag_size_mm": tag_size_mm,
        "margins_px": margins_px,
        "margins_mm": margins,
        "num_x": num_x,
        "num_y": num_y,
        "gap_x_px": gap_x,
        "gap_y_px": gap_y,
        "DPI": dpi,
    }
    points_list = []
    for y in range(num_y):
        if y == 0 or y == num_y - 1:  # first and last row shoud be full
            for x in range(num_x):
                point = np.array(
                    [
                        [x * (tag_size_px + gap_x), y * (tag_size_px + gap_y), 0],
                        [
                            x * (tag_size_px + gap_x) + tag_size_px,
                            y * (tag_size_px + gap_y),
                            0,
                        ],
                        [
                            x * (tag_size_px + gap_x) + tag_size_px,
                            y * (tag_size_px + gap_y) + tag_size_px,
                            0,
                        ],
                        [
                            x * (tag_size_px + gap_x),
                            y * (tag_size_px + gap_y) + tag_size_px,
                            0,
                        ],
                    ],
                    dtype=np.float32,
                )

                point_dict = {
                    "id": idx,
                    "pts": point.tolist(),
                }
                points_list.append(point_dict)

                obj_points.append(point)
                ids.append(idx)
                idx += 1
        else:  # other rows should have only first and last tag
            point = np.array(
                [
                    [0, y * (tag_size_px + gap_y), 0],
                    [tag_size_px, y * (tag_size_px + gap_y), 0],
                    [tag_size_px, y * (tag_size_px + gap_y) + tag_size_px, 0],
                    [0, y * (tag_size_px + gap_y) + tag_size_px, 0],
                ],
                dtype=np.float32,
            )
            point_dict = {
                "id": idx,
                "pts": point.tolist(),
            }
            points_list.append(point_dict)

            obj_points.append(point)
            ids.append(idx)
            idx += 1
            point = np.array(
                [
                    [paper_x_px - tag_size_px, y * (tag_size_px + gap_y), 0],
                    [paper_x_px, y * (tag_size_px + gap_y), 0],
                    [paper_x_px, y * (tag_size_px + gap_y) + tag_size_px, 0],
                    [
                        paper_x_px - tag_size_px,
                        y * (tag_size_px + gap_y) + tag_size_px,
                        0,
                    ],
                ],
                dtype=np.float32,
            )
            point_dict = {
                "id": idx,
                "pts": point.tolist(),
            }
            points_list.append(point_dict)

            obj_points.append(point)
            ids.append(idx)
            idx += 1

    obj_points = np.array(obj_points, dtype=np.float32)
    ids = np.array(ids)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.Board(obj_points, dictionary, ids)

    output_dict["points"] = points_list

    return (
        board,
        (int(paper_size[0] * dpi / 25.4), int(paper_size[1] * dpi / 25.4)),
        output_dict,
    )


def working():
    """
    Returns a pattern of the ArUco markers.
    """
    # p1 = np.array(
    #     [[0, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0], [0.1, 0, 0]], dtype=np.float32
    # )
    # p2 = np.array([[2, 0, 0], [2, 1, 0], [3, 1, 0], [3, 0, 0]], dtype=np.float32)
    # objPoints = np.array([p1, p2])
    # ids = np.array([0, 1])

    objPoints = []
    for i in range(2):
        for j in range(2):
            objPoints.append(
                np.array(
                    [[i, j, 0], [i, j + 1, 0], [i + 1, j + 1, 0], [i + 1, j, 0]],
                    dtype=np.float32,
                )
            )
    ids = np.array([0, 1, 2, 3])

    objPoints = np.array(objPoints, dtype=np.float32)
    print(objPoints.shape)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.Board(objPoints, dictionary, ids)
    return board


if __name__ == "__main__":
    board, res, points_dict = get_rectangle_pattern(30, 300, "A4", 10)

    with open("points.json", "w") as f:
        json.dump(points_dict, f, indent=2)

    # board = working()
    print(board)
    # img = board.generateImage((500, 500), marginSize=10, borderBits=1)
    print(res)
    img = board.generateImage(res, marginSize=10, borderBits=1)
    print(img.shape)
    cv2.imshow("board", img)
    cv2.imwrite("board.png", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
