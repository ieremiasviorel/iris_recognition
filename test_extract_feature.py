import cv2
import numpy as np

import boundary
import extract_feature
import segment
import unwrap

image_names = [
    "CASIA1/9/009_2_1.jpg",
    "CASIA1/9/009_2_3.jpg",
    "CASIA1/9/009_2_4.jpg",
    "CASIA1/9/009_1_1.jpg"
]

unwrapped_images = []

for image_name in image_names:
    print("UNWRAP IMAGE: " + image_name)
    img = cv2.imread(image_name, 0)
    inner_center, inner_radius = boundary.find_iris_inner_bound(img)
    outer_center, outer_radius = boundary.find_iris_outer_bound(img, inner_center, inner_radius)
    eyelids = boundary.find_eyelids(img)

    segmented_img = segment.segment_iris(img, inner_center, inner_radius, outer_center, outer_radius, eyelids)

    unwrapped_img = unwrap.unwrap_iris(segmented_img, inner_center, inner_radius, outer_center, outer_radius)

    unwrapped_images.append(unwrapped_img)

codes = []

for unwrapped_image in unwrapped_images:
    print("COMPUTE CODE")
    code = extract_feature.extract_features(unwrapped_image)
    codes.append(np.asarray(code, dtype=np.float32))

print(str(len(codes)))

print(np.linalg.norm(codes[0] - codes[1]))
print(np.linalg.norm(codes[0] - codes[2]))
print(np.linalg.norm(codes[1] - codes[2]))

print(np.linalg.norm(codes[0] - codes[3]))
print(np.linalg.norm(codes[1] - codes[3]))
print(np.linalg.norm(codes[2] - codes[3]))
