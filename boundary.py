import itertools

import cv2
import numpy as np


def daugman(center, start_radius, end_radius, img):
    """
    - Daugman operator function
    - return for the given circle center, the radius corresponding to the maximum difference in intensity

    INPUT:
        center -- tuple(x, y), start point
        start_r -- int, start radius
        img -- np.array(), grayscale picture
    OUTPUT:
        max_val - maximum difference in intensity
        [(center_X, center_Y), radius] - center point and radius of circle
    """
    # array of differences between sums of elements of neighboring circles
    tmp = []

    mask = np.zeros_like(img)

    for r in range(start_radius, end_radius):
        # draw circle on mask
        cv2.circle(mask, center, r, 255, 1)
        # isolate image pixels belonging on the specific circle
        radii = img & mask
        # normalize and sum all pixels on circle
        tmp.append(radii[radii > 0].sum() / (2 * np.pi * r))
        # reinitialize mask
        mask.fill(0)

    tmp = np.array(tmp)
    # subtract neighboring values 2-by-2
    tmp = tmp[1:] - tmp[:-1]
    tmp = abs(cv2.GaussianBlur(tmp[:-1], (1, 5), 0))

    max_idx = np.argmax(tmp)
    # return value, center coords, radius
    return tmp[max_idx], [center, max_idx + start_radius]


def find_iris_inner_bound(img):
    """
    INPUT:
        img - graysacale image as np.array()
    OUTPUT:
        (center_X, center_Y), radius
    """
    height, width = img.shape
    start_radius = int(width / 10)
    end_radius = 2 * start_radius
    coord_range = range(int(width / 4), int(3 * width / 4), 4)
    all_points = list(itertools.product(coord_range, coord_range))

    values = []
    coords = []

    for p in all_points:
        val, circle = daugman(p, start_radius, end_radius, img)
        values.append(val)
        coords.append(circle)
    return coords[np.argmax(values)]


def find_iris_outer_bound(img, inner_center, inner_radius):
    """
    INPUT:
        img - graysacale image as np.array()
        inner_center - center of the pupil circle
        inner_radius - radius of the pupil circle
    OUTPUT:
        (center_X, center_Y), radius
    """
    height, width = img.shape
    start_radius = int(inner_radius + width / 10)
    end_radius = int(width / 3 + 20)
    coord_range_x = range(int(inner_center[0] - 10), int(inner_center[0] + 10), 4)
    coord_range_y = range(int(inner_center[1] - 10), int(inner_center[1] + 10), 4)
    all_points = list(itertools.product(coord_range_x, coord_range_y))

    values = []
    coords = []

    for p in all_points:
        val, circle = daugman(p, start_radius, end_radius, img)
        values.append(val)
        coords.append(circle)

    return coords[np.argmax(values)]


def find_eyelids(img):
    """
    INPUT:
        img - graysacale image as np.array()
    OUTPUT:
        [[center_X, center_Y, radius]]
    """
    '''
    blurred = cv2.morphologyEx(img, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    blurred = cv2.morphologyEx(blurred, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    blurred = cv2.medianBlur(blurred, 7)
    '''

    blurred_img = cv2.medianBlur(img, 7)
    circles = cv2.HoughCircles(blurred_img, cv2.HOUGH_GRADIENT, 1, 200,
                               param1=50, param2=30, minRadius=150, maxRadius=600)
    if circles is None:
        return []
    circles = np.uint16(np.around(circles))
    return circles.reshape((circles.shape[1], circles.shape[2]))


def find_eyelids_daugman(img, outer_center, outer_radius):
    """
        INPUT:
            img - graysacale image as np.array()
            inner_center - center of the pupil circle
            inner_radius - radius of the pupil circle
        OUTPUT:
            (center_X, center_Y), radius
    """
    height, width = img.shape
    start_r = int(outer_radius + width / 10)
    coord_range_x = range(
        int(outer_center[0] - 200), int(outer_center[0] + 200), 4)
    coord_range_y = range(
        int(outer_center[1] - 10), int(outer_center[1] + 10), 4)
    all_points = list(itertools.product(coord_range_x, coord_range_y))

    values = []
    coords = []

    for p in all_points:
        val, circle = daugman(p, start_r, img)
        values.append(val)
        coords.append(circle)
    return coords[np.argmax(values)]
