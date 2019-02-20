import sys
import numpy as np
from math import floor
from cv2 import bitwise_and, imread, imshow, circle, waitKey, destroyAllWindows, logPolar, WARP_FILL_OUTLIERS

from boundary import find_iris_inner_bound, find_iris_outer_bound

img_filename = sys.argv[1]
img = imread(img_filename, 0)

inner_center, inner_radius = find_iris_inner_bound(img)
outer_center, outer_radius = find_iris_outer_bound(img, inner_center, inner_radius)

circle(img, inner_center, inner_radius, (255, 255, 255), 1)
circle(img, outer_center, outer_radius, (255, 255, 255), 1)

imshow("boundary image", img)

waitKey(0)
destroyAllWindows()