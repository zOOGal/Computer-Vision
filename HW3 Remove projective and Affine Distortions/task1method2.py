import myFunctions
import cv2 as cv2
import pdb

image1 = cv2.imread('Img1.jpg')
image2 = cv2.imread('Img2.jpeg')
image3 = cv2.imread('Img3.jpg')

img1 = list()
img1.append([162, 66, 1])
img1.append([91, 670, 1])
img1.append([1048, 757, 1])
img1.append([1019, 346, 1])

img2 = list()
img2.append([425, 231, 1])
img2.append([424, 345, 1])
img2.append([514, 298, 1])
img2.append([512, 167, 1])


img3 = list()
img3.append([2092, 736, 1])
img3.append([2121, 1427, 1])
img3.append([2674, 1299, 1])
img3.append([2646, 747, 1])


# ============================Twp-step Method=============================
vl1 = myFunctions.v_line(img1[0], img1[1], img1[2], img1[3])
h_1 = myFunctions.homography_vline_back(vl1)
result1 = myFunctions.mapping(h_1, image1)
cv2.imwrite('affine1.jpg', result1)

img_affine1 = cv2.imread('affine1.jpg')
h_affine_1 = myFunctions.homography_affine(h_1, img1)
result1_affine = myFunctions.mapping(h_affine_1, img_affine1)
cv2.imwrite('normal1.jpg', result1_affine)

vl2 = myFunctions.v_line(img2[0], img2[1], img2[2], img2[3])
h_2 = myFunctions.homography_vline_back(vl2)
result2 = myFunctions.mapping(h_2, image2)
cv2.imwrite('affine2.jpg', result2)

img2_affine = cv2.imread('affine2.jpeg')
h_affine_2 = myFunctions.homography_affine(h_2, img2)
result2_affine = myFunctions.mapping(h_affine_2, image2)
cv2.imwrite('normal2.jpg', result2_affine)

# ===========================image 3===================================
vl3 = myFunctions.v_line(img3[0], img3[1], img3[2], img3[3])
h_3 = myFunctions.homography_vline_back(vl3)
result3 = myFunctions.mapping(h_3, image3)
cv2.imwrite('affine3.jpg', result3)

img_affine3 = cv2.imread('affine3.jpg')
h_affine_3 = myFunctions.homography_affine(h_3, img3)
result3_affine = myFunctions.mapping(h_affine_3, img_affine3)
cv2.imwrite('normal3.jpg', result3_affine)

pdb.set_trace()

