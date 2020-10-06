import myFunctions
import cv2 as cv2
import pdb

image1 = cv2.imread('building.jpg')
image2 = cv2.imread('window.jpg')

img1 = list()
img1.append([90, 210, 1])
img1.append([63, 382, 1])
img1.append([285, 347, 1])
img1.append([307, 162, 1])

img2 = list()
img2.append([131, 59, 1])
img2.append([178, 790, 1])
img2.append([483, 717, 1])
img2.append([459, 126, 1])

# ============================image 1=============================
vl1 = myFunctions.v_line(img1[0], img1[1], img1[2], img1[3])
h_1 = myFunctions.homography_vline_back(vl1)
result1 = myFunctions.mapping(h_1, image1)
cv2.imwrite('affine_building.jpg', result1)

img_affine1 = cv2.imread('affine_building.jpg')
h_affine_1 = myFunctions.homography_affine(h_1, img1)
result1_affine = myFunctions.mapping(h_affine_1, img_affine1)
cv2.imwrite('normal_building.jpg', result1_affine)

# ===========================image2================================
vl2 = myFunctions.v_line(img2[0], img2[1], img2[2], img2[3])
h_2 = myFunctions.homography_vline_back(vl2)
result2 = myFunctions.mapping(h_2, image2)
cv2.imwrite('affine_window.jpg', result2)

img2_affine = cv2.imread('affine_window.jpeg')
h_affine_2 = myFunctions.homography_affine(h_2, img2)
result2_affine = myFunctions.mapping(h_affine_2, image2)
cv2.imwrite('normal_window.jpg', result2_affine)


pdb.set_trace()

