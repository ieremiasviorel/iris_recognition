import itertools
import cv2
import numpy as np


def daugman(center, start_r, img):
    """
    - Daugman operator function
    - return for the given circle center, the radius corresponding to the maximum difference in intensity

    INPUT:
        center -- tuple(x, y)
        start_r -- int
        img -- grayscale picture as np.array()
    OUTPUT:
        max_val - maximum difference in intensity
        [(center_X, center_Y), radius]
    """
    height, width = img.shape
    # array of differences between sums of elements of neighboring circles
    tmp = []

    mask = np.zeros_like(img)

    for r in range(start_r, int(width / 3 + 20)):
        # draw circle on mask
        cv2.circle(mask, center, r, 255, 1)
        # isolate image pixels belonging on the specific circle
        radii = img & mask
        # normalize and sum all pixels on circle
        tmp.append(radii[radii > 0].sum()/(2 * np.pi * r))
        # reinitialize mask
        mask.fill(0)

    tmp = np.array(tmp)
    # subtract neighboring values 2-by-2
    tmp = tmp[1:] - tmp[:-1]
    tmp = abs(cv2.GaussianBlur(tmp[:-1], (1, 5), 0))

    max_idx = np.argmax(tmp)
    # return value, center coords, radius
    return tmp[max_idx], [center, max_idx + start_r]


def find_iris_inner_bound(img):
    """
    INPUT:
        img -- graysacale image as np.array()

    OUTPUT:
        ((center_X, center_Y), radius)
    """
    height, width = img.shape
    start_r = int(width / 10)
    coord_range = range(int(width / 3), int(2 * width / 3), 4)
    all_points = list(itertools.product(coord_range, coord_range))

    values = []
    coords = []

    for p in all_points:
        tmp = daugman(p, start_r, img)
        if tmp is not None:
            val, circle = tmp
            values.append(val)
            coords.append(circle)
    return coords[np.argmax(values)]


def find_iris_outer_bound(img, inner_center, inner_radius):
    """
    INPUT:
        img -- graysacale image as np.array()
        inner_center - center of the pupil circle
        inner_radius - radius of the pupil circle

    OUTPUT:
        ((center_X, center_Y), radius)
    """
    height, width = img.shape
    start_r = int(inner_radius + width / 10)
    coord_range_x = range(
        int(inner_center[0] - 10), int(inner_center[0] + 10), 4)
    coord_range_y = range(
        int(inner_center[1] - 10), int(inner_center[1] + 10), 4)
    # coord_range_x = range(
    #    int(inner_center[0]), int(inner_center[0] + 1), 4)
    # coord_range_y = range(
    #    int(inner_center[1]), int(inner_center[1] + 1), 4)
    all_points = list(itertools.product(coord_range_x, coord_range_y))

    values = []
    coords = []

    for p in all_points:
        tmp = daugman(p, start_r, img)
        if tmp is not None:
            val, circle = tmp
            values.append(val)
            coords.append(circle)
    return coords[np.argmax(values)]
