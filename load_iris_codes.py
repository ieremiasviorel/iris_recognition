import pickle
import random

import cv2
import numpy as np
from scipy.spatial import distance


def compute_iris_code_distance(iris_code_1, iris_code_2):
    i1 = np.uint16(np.around(iris_code_1))
    i2 = np.uint16(np.around(iris_code_2))
    return distance.hamming(i1, i2)
    '''
    iris_code_1 = np.asarray(iris_code_1, dtype=np.float32)
    iris_code_2 = np.asarray(iris_code_2, dtype=np.float32)
    return np.linalg.norm(iris_code_1 - iris_code_2)
    '''


def get_random_eye_picture(subject_index):
    random_subject_index = subject_index
    while (random_subject_index == subject_index):
        random_subject_index = random.randrange(108)
    return [random_subject_index, random.randrange(7)]


iris_codes = []

for subject_index in range(1, 109):
    f = open("iris_codes/" + str(subject_index).zfill(3) + ".pkl", "rb")
    iris_codes.append(pickle.load(f))
    f.close()
same_eye_distances = []

diff_eye_distances = []

for subject_index, iris_codes_per_eye in enumerate(iris_codes):
    same_eye_combos = [
        [0, 1],
        [1, 2],
        [0, 2],
        [3, 4],
        [4, 5],
        [5, 6]
    ]

    for combo in same_eye_combos:
        dist = compute_iris_code_distance(iris_codes_per_eye[combo[0]], iris_codes_per_eye[combo[1]])
        if dist > 100:
            print("DIFF: " + str(dist) + " - " + str(subject_index) + " | " + str(combo))
        same_eye_distances.append(dist)

    diff_eye_combos = [[i, get_random_eye_picture(subject_index)] for i in range(6)]

    for combo in diff_eye_combos:
        dist = compute_iris_code_distance(iris_codes_per_eye[combo[0]], iris_codes[combo[1][0]][combo[1][1]])
        diff_eye_distances.append(dist)

print(np.mean(same_eye_distances))
print(np.std(same_eye_distances))
print(np.amin(same_eye_distances))
print(np.amax(same_eye_distances))

print()

print(np.mean(diff_eye_distances))
print(np.std(diff_eye_distances))
print(np.amin(diff_eye_distances))
print(np.amax(diff_eye_distances))
