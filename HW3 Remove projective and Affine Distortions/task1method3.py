import myFunctions
import cv2 as cv2
import pdb

image1 = cv2.imread('Img1.jpg')
image2 = cv2.imread('Img2.jpeg')
image3 = cv2.imread('Img3.jpg')

img1_set1 = list()
img1_set1.append([163, 69, 1])
img1_set1.append([91, 670, 1])
img1_set1.append([1048, 757, 1])
img1_set1.append([1017, 348, 1])
img1_set2 = list()

img1_set2.append([92, 674, 1])
img1_set2.append([68, 872, 1])
img1_set2.append([223, 875, 1])
img1_set2.append([240, 686, 1])

img2_set1 = list()

img2_set1.append([479, 720, 1])
img2_set1.append([481, 872, 1])
img2_set1.append([606, 922, 1])
img2_set1.append([600, 738, 1])
img2_set2 = list()

img2_set2.append([383, 576, 1])
img2_set2.append([381, 695, 1])
img2_set2.append([463, 705, 1])
img2_set2.append([461, 566, 1])

img3_set1 = list()
img3_set1.append([2092, 736, 1])
img3_set1.append([2121, 1427, 1])
img3_set1.append([2674, 1299, 1])
img3_set1.append([2646, 747, 1])
img3_set2 = list()
img3_set2.append([680, 750, 1])
img3_set2.append([735, 2074, 1])
img3_set2.append([1769, 1607, 1])
img3_set2.append([1767, 790, 1])

h_1step_1 = myFunctions.onestep(img1_set1, img1_set2)
result1 = myFunctions.mapping(h_1step_1, image1)
cv2.imwrite('onestep1.jpg', result1)

h_1step_2 = myFunctions.onestep(img2_set1, img2_set2)
result2 = myFunctions.mapping(h_1step_2, image2)
cv2.imwrite('onestep2.jpg', result2)

h_1step_3 = myFunctions.onestep(img3_set1, img3_set2)
result3 = myFunctions.mapping(h_1step_3, image3)
cv2.imwrite('onestep3.jpg', result3)

pdb.set_trace()