import cv2
import numpy as np

from boundary import find_iris_inner_bound, find_iris_outer_bound


def segment_iris(img, inner_center, inner_radius, outer_center, outer_radius):
    """
    INPUT:
        img -- graysacale image as np.array()
        inner_center - (x,y) coordinates of pupil circle
        inner_radius - radius of pupil circle
        outer_center - (x, y) coordinates of iris circle
        outer_radius - radius of iris circle
    OUTPUT:
        segmented_img - image of the iris only
    """
    inner_center, inner_radius = find_iris_inner_bound(img)
    outer_center, outer_radius = find_iris_outer_bound(
        img, inner_center, inner_radius)

    mask = np.zeros(img.shape, np.uint8)
    cv2.circle(mask, outer_center, outer_radius, 255, thickness=-1)
    cv2.circle(mask, inner_center, inner_radius, 0, thickness=-1)
    segmented_img = np.bitwise_and(img, mask)

    #segmented_img = segmented_img[(outer_center[1] - outer_radius): (outer_center[1] + outer_radius + 1),
    #                              (outer_center[0] - outer_radius): (outer_center[0] + outer_radius + 1)]

    return segmented_img
