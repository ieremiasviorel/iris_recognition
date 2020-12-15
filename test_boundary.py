import cv2

import boundary

img_filename = "sample1.jpg"
img = cv2.imread(img_filename, 0)

inner_center, inner_radius = boundary.find_iris_inner_bound(img)
outer_center, outer_radius = boundary.find_iris_outer_bound(img, inner_center, inner_radius)
eyelids = boundary.find_eyelids(img)

cv2.circle(img, outer_center, outer_radius, (255, 255, 255), 1)
cv2.circle(img, inner_center, inner_radius, (255, 255, 255), 1)
for eyelid in eyelids:
    cv2.circle(img, (eyelid[0], eyelid[1]), eyelid[2], (255, 255, 255), 1)

cv2.imshow("boundary image", img)

print(inner_center, inner_radius)
print(outer_center, outer_radius)

cv2.waitKey(0)
cv2.destroyAllWindows()
