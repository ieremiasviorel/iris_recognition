import cv2

import boundary
import boundary_point

img_filename = "sample1.jpg"
img = cv2.imread(img_filename, 0)

inner_y, inner_x, inner_radius = boundary_point.searchInnerBound(img)
outer_y, outer_x, outer_radius = boundary_point.searchOuterBound(img, inner_y, inner_x, inner_radius)
eyelids = boundary.find_eyelids(img)

cv2.circle(img, (int(outer_x), int(outer_y)), outer_radius, (255, 255, 255), 1)
cv2.circle(img, (int(inner_x), int(inner_y)), inner_radius, (255, 255, 255), 1)
for eyelid in eyelids:
    cv2.circle(img, (eyelid[0], eyelid[1]), eyelid[2], (255, 255, 255), 1)

cv2.imshow("boundary image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
