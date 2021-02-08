import sys
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np

from boundary import find_iris_inner_bound, find_iris_outer_bound
from segment import segment_iris
from unwrap import unwrap_iris
from extract_feature import extract_features

root = tk.Tk()
root.withdraw()

file_path_1 = filedialog.askopenfilename()
img_1 = cv2.imread(file_path_1, 0)

inner_center, inner_radius = find_iris_inner_bound(img_1)
outer_center, outer_radius = find_iris_outer_bound(
    img_1, inner_center, inner_radius)

segmented_img = segment_iris(
    img_1, inner_center, inner_radius, outer_center, outer_radius)

unwrapped_img = unwrap_iris(
    segmented_img, inner_center, inner_radius, outer_center, outer_radius)

_, iris_code_1 = extract_features(unwrapped_img)

file_path_2 = filedialog.askopenfilename()
img_2 = cv2.imread(file_path_2, 0)

inner_center, inner_radius = find_iris_inner_bound(img_2)
outer_center, outer_radius = find_iris_outer_bound(
    img_2, inner_center, inner_radius)

segmented_img = segment_iris(
    img_2, inner_center, inner_radius, outer_center, outer_radius)

unwrapped_img = unwrap_iris(
    segmented_img, inner_center, inner_radius, outer_center, outer_radius)

_, iris_code_2 = extract_features(unwrapped_img)

iris_code_1 = np.asarray(iris_code_1, dtype=np.float32)
iris_code_2 = np.asarray(iris_code_2, dtype=np.float32)

iris_code_dist = np.linalg.norm(iris_code_1 - iris_code_2)
print(iris_code_dist)
if iris_code_dist < 60:
    print("YES")
else:
    print("NO")