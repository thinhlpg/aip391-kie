import numpy as np
import cv2
import math
from scipy import ndimage

img_path = r'../../data/experiments/preprocess-sample/mcocr_public_145013unjxn.jpg'
img = cv2.imread(img_path)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
lines = lines.reshape(-1,4)
print(lines)
angles = []
line_lengths = []

for x1, y1, x2, y2 in lines:
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)
    #line_length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    #line_lengths.append(line_length)

print(angles)
print((line_lengths))
median_angle = np.median(angles)
img_rotated = ndimage.rotate(img, median_angle)
cv2.imshow('', img_rotated)
cv2.waitKey(0)

print("Angle is {}".format(median_angle))
# cv2.imwrite('rotated_2.jpg', img_rotated)