import cv2

import boundary
import extract_feature
import segment
import unwrap

image_name = "sample1.jpg"
img = cv2.imread(image_name, 0)

inner_center, inner_radius = boundary.find_iris_inner_bound(img)
outer_center, outer_radius = boundary.find_iris_outer_bound(img, inner_center, inner_radius)
eyelids = boundary.find_eyelids(img)

segmented_img = segment.segment_iris(img, inner_center, inner_radius, outer_center, outer_radius, eyelids)

unwrapped_img = unwrap.unwrap_iris(segmented_img, inner_center, inner_radius, outer_center, outer_radius)

filtered_images, code = extract_feature.extract_features(unwrapped_img)

for i in range(len(filtered_images)):
    cv2.imwrite("sample1_" + str(i + 1) + ".jpg", filtered_images[i])
