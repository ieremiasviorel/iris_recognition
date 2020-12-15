import cv2

import boundary
import segment
import unwrap

img_filename = "sample4.jpg"
img = cv2.imread(img_filename, 0)

inner_center, inner_radius = boundary.find_iris_inner_bound(img)
outer_center, outer_radius = boundary.find_iris_outer_bound(img, inner_center, inner_radius)
eyelids = boundary.find_eyelids(img)

segmented_img = segment.segment_iris(img, inner_center, inner_radius, outer_center, outer_radius, eyelids)

unwrapped_img = unwrap.unwrap_iris(segmented_img, inner_center, inner_radius, outer_center, outer_radius)

cv2.circle(img, inner_center, inner_radius, (255, 255, 255), 1)
cv2.circle(img, outer_center, outer_radius, (255, 255, 255), 1)
for eyelid in eyelids:
    cv2.circle(img, (eyelid[0], eyelid[1]), eyelid[2], (255, 255, 255), 1)

cv2.imwrite("sample4.unwrap.jpg", unwrapped_img)

cv2.imshow("original image", img)
cv2.imshow("segmented_image", segmented_img)
cv2.imshow("unwrapped_image", unwrapped_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
