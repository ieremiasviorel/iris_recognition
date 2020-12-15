import numpy as np
import cv2 as cv

img = cv.imread('sample3.jpg', 0)
img = cv.medianBlur(img, 5)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

"""
    INNER CIRCLE - GREEN
"""
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 200,
                          param1=50, param2=30, minRadius=10, maxRadius=50)
print("INNER CIRCLE: " + str(circles.shape))
print(circles)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 1)
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (0, 255, 0), 2)

"""
    OUTER CIRCLE - RED
"""
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 200,
                          param1=50, param2=30, minRadius=75, maxRadius=125)
print("OUTER CIRCLE: " + str(circles.shape))
print(circles)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 0, 255), 1)
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 2)

"""
    BOTTOM EYE LID - BLUE
"""
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 200,
                          param1=50, param2=30, minRadius=150, maxRadius=600)
print("BOTTOM EYE LID CIRCLE: " + str(circles.shape))
print(circles)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (255, 0, 0), 1)
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (255, 0, 0), 2)

"""
    TOP EYE LID - CYAN
"""
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 200,
                          param1=50, param2=30, minRadius=150, maxRadius=600)
print("TOP EYE LID CIRCLE: " + str(circles.shape))
circles = np.uint16(np.around(circles))
print(circles)
for i in circles[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (255, 255, 0), 1)
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (255, 255, 0), 2)

cv.imshow('detected circles', cimg)
cv.waitKey(0)
cv.destroyAllWindows()
