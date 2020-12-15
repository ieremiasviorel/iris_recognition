import functools

import cv2
import numpy as np


def segment_iris(img, inner_center, inner_radius, outer_center, outer_radius, eyelids):
    """
    INPUT:
        img - graysacale image as np.array()
        inner_center - (x,y) coordinates of pupil circle
        inner_radius - radius of pupil circle
        outer_center - (x, y) coordinates of iris circle
        outer_radius - radius of iris circle
    OUTPUT:
        segmented_img - image of the iris only
    """
    empty = np.zeros(img.shape, np.uint8)

    masks = [cv2.circle(empty.copy(), (i[0], i[1]), i[2], 255, thickness=-1) for i in eyelids]
    masks.append(cv2.circle(empty.copy(), outer_center, outer_radius, 255, thickness=-1))

    mask = functools.reduce(lambda a, b: np.bitwise_and(a, b), masks)

    cv2.circle(mask, inner_center, inner_radius, 0, thickness=-1)

    return np.bitwise_and(img, mask)
