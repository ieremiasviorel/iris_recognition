import cv2
import numpy as np


def extractFeature(img):

    ksize = [10, 15, 25]
    sigma = 5
    _lambda = 5
    theta = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    gamma = 0.2
    psi = 0
    ktype = cv2.CV_32F

    filtered_images = []
    code = []

    for _ksize in ksize:
        for _theta in theta:
            kern = cv2.getGaborKernel((_ksize, _ksize), sigma,
                                      _theta, _lambda, gamma, psi, ktype)
            kern /= 1.5*kern.sum()
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            filtered_images.append(fimg)

    for img_idx in range(8):
        for sub_img_idx in range(8):
            sub_img = filtered_images[img_idx][:, 64 *
                                               sub_img_idx: 64 * (sub_img_idx + 1)]
            mean = np.mean(sub_img)
            aad = 1 / (64.0 * 64.0) * np.sum(np.absolute(sub_img - mean))
            code.append(aad)

    return code
