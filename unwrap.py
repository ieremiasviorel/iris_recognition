import cv2
import numpy as np


def unwrap_iris(img, inner_center, inner_radius, outer_center, outer_radius):
    height, width = img.shape
    M = width / np.log(outer_radius)

    unwrapped_img = cv2.logPolar(img, outer_center, M, cv2.WARP_FILL_OUTLIERS)

    center_shift_x = np.absolute(outer_center[0] - inner_center[0])
    center_shift_y = np.absolute(outer_center[1] - inner_center[1])

    if center_shift_x > center_shift_y:
        center_shift = center_shift_x
    else:
        center_shift = center_shift_y

    center_shift += 5

    useful_width = (outer_radius - inner_radius) + center_shift

    unwrapped_img = unwrapped_img[:, (width - useful_width): width]

    scaled_img = cv2.resize(unwrapped_img, (64, 512))

    rotated_img = cv2.rotate(scaled_img, cv2.ROTATE_90_CLOCKWISE)

    return rotated_img
