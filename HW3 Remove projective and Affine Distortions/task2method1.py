import cv2 as cv2
import myFunctions
import numpy as np
import pdb

image1 = cv2.imread('building.jpg')
image2 = cv2.imread('window.jpg')

img1 = list()
img1.append([214, 184, 1])
img1.append([189, 361, 1])
img1.append([285, 347, 1])
img1.append([307, 162, 1])

img2 = list()
img2.append([131, 59, 1])
img2.append([178, 790, 1])
img2.append([483, 717, 1])
img2.append([459, 126, 1])

undistorted1 = list()
undistorted1.append([0, 0, 1])
undistorted1.append([0, 50, 1])
undistorted1.append([30, 50, 1])
undistorted1.append([30, 0, 1])     # unit is 50 dm to avoid calculation mess

undistorted2 = list()
undistorted2.append([0, 0, 1])
undistorted2.append([0, 100, 1])
undistorted2.append([150, 100, 1])
undistorted2.append([150, 0, 1])

# =====================point to point(p2p) method===============================
h_1 = myFunctions.cal_homography(img1, undistorted1)
p2p_1 = myFunctions.mapping(h_1, image1)
cv2.imwrite('p2p_building.jpg', p2p_1)

h_2 = myFunctions.cal_homography(img2, undistorted2)
p2p_2 = myFunctions.mapping(h_2, image2)
cv2.imwrite('p2p_window.jpg', p2p_2)


pdb.set_trace()
