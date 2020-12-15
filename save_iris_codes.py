import multiprocessing
import pickle

import cv2
import numpy as np

import boundary
import extract_feature
import segment
import unwrap


def get_image_filename(subject_index, sample_index):
    return "CASIA1/" + \
           str(subject_index) + "/" + \
           str(subject_index).zfill(3) + "_" + str(sample_index // 4 + 1) + "_" + \
           str(sample_index - 3 * (sample_index // 4)) + \
           ".jpg"


def get_image_iris_code(img_filename):
    img = cv2.imread(img_filename, 0)

    inner_center, inner_radius = boundary.find_iris_inner_bound(img)
    outer_center, outer_radius = boundary.find_iris_outer_bound(img, inner_center, inner_radius)
    eyelids = boundary.find_eyelids(img)

    segmented_img = segment.segment_iris(img, inner_center, inner_radius, outer_center, outer_radius, eyelids)

    unwrapped_img = unwrap.unwrap_iris(segmented_img, inner_center, inner_radius, outer_center, outer_radius)

    iris_code = extract_feature.extract_features(unwrapped_img)

    return np.asarray(iris_code, dtype=np.float32)


def compute_and_save_iris_code(subject_index):
    print("SUBJECT: " + str(subject_index))
    iris_codes = []
    for sample_index in range(1, 8):
        print("SAMPLE: " + str(subject_index) + "/" + str(sample_index))
        img_filename = get_image_filename(subject_index, sample_index)
        iris_code = get_image_iris_code(img_filename)
        iris_codes.append(iris_code)
    f = open("iris_codes/" + str(subject_index).zfill(3) + ".pkl", "wb")
    pickle.dump(iris_codes, f)
    f.close()


"""
for subject_index in range(1, 109):
    print("SUBJECT: " + str(subject_index))
    iris_codes = []
    for sample_index in range(1, 8):
        print("SAMPLE: " + str(subject_index) + "/" + str(sample_index))
        img_filename = get_image_filename(subject_index, sample_index)
        iris_code = get_image_iris_code(img_filename)
        iris_codes.append(iris_code)
    f = open("iris_codes/" + str(subject_index).zfill(3) + ".pkl", "wb")
    pickle.dump(iris_codes, f)
    f.close()
"""

if __name__ == '__main__':
    pool = multiprocessing.Pool(16)
    zip(*pool.map(compute_and_save_iris_code, range(1, 109)))
