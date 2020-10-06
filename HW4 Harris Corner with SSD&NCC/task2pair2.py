import cv2 as cv2
import harrisConner
import siftMatch

cfl1 = cv2.imread('cfl1.jpg')
cfl2 = cv2.imread('cfl2.jpg')

sigma = 3

cfl1_corner = harrisConner.harris_corner(cfl1, sigma)
cfl2_corner = harrisConner.harris_corner(cfl2, sigma)

if len(cfl1_corner) < len(cfl2_corner):
    cfl_pair1 = harrisConner.find_local_match(cfl1, cfl2, cfl1_corner, cfl2_corner, 21, mode='SSD')
    cfl_finish1 = harrisConner.mark_pair(cfl1, cfl2, cfl_pair1)

    cfl_pair2 = harrisConner.find_local_match(cfl1, cfl2, cfl1_corner, cfl2_corner, 21, mode='NCC')
    cfl_finish2 = harrisConner.mark_pair(cfl1, cfl2, cfl_pair2)
else:
    cfl_pair1 = harrisConner.find_local_match(cfl2, cfl1, cfl2_corner, cfl1_corner, 21, mode='SSD')
    cfl_finish1 = harrisConner.mark_pair(cfl2, cfl1, cfl_pair1)

    cfl_pair2 = harrisConner.find_local_match(cfl2, cfl1, cfl2_corner, cfl1_corner, 21, mode='NCC')
    cfl_finish2 = harrisConner.mark_pair(cfl2, cfl1, cfl_pair2)

cv2.imwrite('cfl_compare_ssd_%s.jpg' % sigma, cfl_finish1)
cv2.imwrite('cfl_compare_ncc_%s.jpg' % sigma, cfl_finish2)

cfl_sift = siftMatch.sift_match(cfl1, cfl2, max_show=100, mode='knn')
cv2.imwrite('cfl_sift.jpg', cfl_sift)
