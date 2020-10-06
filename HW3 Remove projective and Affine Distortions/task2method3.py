import myFunctions
import cv2 as cv2

image1 = cv2.imread('building.jpg')
image2 = cv2.imread('window.jpg')

img1_set1 = list()
img1_set1.append([212, 184, 1])
img1_set1.append([188, 361, 1])
img1_set1.append([286, 347, 1])
img1_set1.append([306, 159, 1])
img1_set2 = list()
img1_set2.append([94, 215, 1])
img1_set2.append([64, 383, 1])
img1_set2.append([149, 371, 1])
img1_set2.append([177, 194, 1])


img2_set1 = list()
img2_set1.append([710, 243, 1])
img2_set1.append([723, 690, 1])
img2_set1.append([934, 637, 1])
img2_set1.append([929, 270, 1])

img2_set2 = list()

img2_set2.append([67, 49, 1])
img2_set2.append([133, 868, 1])
img2_set2.append([522, 761, 1])
img2_set2.append([501, 131, 1])

h_1step_1 = myFunctions.onestep(img1_set1, img1_set2)
result1 = myFunctions.mapping(h_1step_1, image1)
cv2.imwrite('onestep_building.jpg', result1)
#
h_1step_2 = myFunctions.onestep(img2_set1, img2_set2)
result2 = myFunctions.mapping(h_1step_2, image2)
cv2.imwrite('onestep_window.jpg', result2)
