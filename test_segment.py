import sys
import cv2

from boundary import find_iris_inner_bound, find_iris_outer_bound
from segment import segment_iris

img_filename = sys.argv[1]
img = cv2.imread(img_filename, 0)

inner_center, inner_radius = find_iris_inner_bound(img)
outer_center, outer_radius = find_iris_outer_bound(img, inner_center, inner_radius)

segmented_img = segment_iris(img, inner_center, inner_radius, outer_center, outer_radius)

cv2.circle(img, inner_center, inner_radius, (255, 255, 255), 1)
cv2.circle(img, outer_center, outer_radius, (255, 255, 255), 1)

cv2.imshow("original image", img)
cv2.imshow("segmented_image", segmented_img)

cv2.waitKey(0)
cv2.destroyAllWindows()