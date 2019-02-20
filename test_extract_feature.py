import sys
import cv2
import numpy as np

from boundary import find_iris_inner_bound, find_iris_outer_bound
from segment import segment_iris
from unwrap import unwrap_iris
from extract_feature import extractFeature

img_filename = sys.argv[1]
img = cv2.imread(img_filename, 0)

inner_center, inner_radius = find_iris_inner_bound(img)
outer_center, outer_radius = find_iris_outer_bound(
    img, inner_center, inner_radius)

segmented_img = segment_iris(
    img, inner_center, inner_radius, outer_center, outer_radius)

unwrapped_img = unwrap_iris(
    segmented_img, inner_center, inner_radius, outer_center, outer_radius)

cv2.circle(img, inner_center, inner_radius, (255, 255, 255), 1)
cv2.circle(img, outer_center, outer_radius, (255, 255, 255), 1)

#cv2.imshow("original image", img)
#cv2.imshow("segmented_image", segmented_img)
cv2.imshow("unwrapped_image", unwrapped_img)

iris_code = extractFeature(unwrapped_img)
print(iris_code)

cv2.waitKey(0)
cv2.destroyAllWindows()

code_1 = np.asarray(extractFeature(cv2.imread('009_2_1.jpg')), dtype=np.float32)
code_2 = np.asarray(extractFeature(cv2.imread('009_2_3.jpg')), dtype=np.float32)
code_3 = np.asarray(extractFeature(cv2.imread('009_2_4.jpg')), dtype=np.float32)
code_4 = np.asarray(extractFeature(cv2.imread('test2.jpg')), dtype=np.float32)

print(np.linalg.norm(code_1 - code_2))
print(np.linalg.norm(code_1 - code_3))
print(np.linalg.norm(code_2 - code_3))

print(np.linalg.norm(code_1 - code_4))
print(np.linalg.norm(code_2 - code_4))
print(np.linalg.norm(code_3 - code_4))