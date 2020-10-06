import cv2 as cv2
import myFunctions

image1 = cv2.imread('Img1.jpg')
image2 = cv2.imread('Img2.jpeg')
image3 = cv2.imread('Img3.jpg')

img1 = list()
img1.append([367, 578, 1])
img1.append([364, 619, 1])
img1.append([395, 623, 1])
img1.append([396, 582, 1])

img2 = list()
img2.append([476, 716, 1])
img2.append([480, 870, 1])
img2.append([606, 923, 1])
img2.append([601, 736, 1])


img3 = list()
img3.append([2058, 698, 1])
img3.append([2090, 1476, 1])
img3.append([2693, 1327, 1])
img3.append([2665, 718, 1])

undistorted1 = list()
undistorted1.append([0, 0, 1])
undistorted1.append([0, 85, 1])
undistorted1.append([75, 85, 1])
undistorted1.append([75, 0, 1])

undistorted2 = list()
undistorted2.append([0, 0, 1])
undistorted2.append([0, 74, 1])
undistorted2.append([84, 74, 1])
undistorted2.append([84, 0, 1])

undistorted3 = list()
undistorted3.append([0, 0, 1])
undistorted3.append([0, 55, 1])
undistorted3.append([36, 55, 1])
undistorted3.append([36, 0, 1])

# =====================point to point(p2p) method===============================
h_1 = myFunctions.cal_homography(img1, undistorted1)
p2p_1 = myFunctions.mapping(h_1, image1)
cv2.imwrite('p2p_1.jpg', p2p_1)

h_2 = myFunctions.cal_homography(img2, undistorted2)
p2p_2 = myFunctions.mapping(h_2, image2)
cv2.imwrite('p2p_2.jpg', p2p_2)

h_3 = myFunctions.cal_homography(img3, undistorted3)
p2p_3 = myFunctions.mapping(h_3, image3)
cv2.imwrite('p2p_3.jpg', p2p_3)


