import cv2 as cv2
import harrisConner
import siftMatch

sea1 = cv2.imread('sea1.jpg')
sea2 = cv2.imread('sea2.jpg')

sigma = 3

sea1_corner = harrisConner.harris_corner(sea1, sigma)
sea1_harris_finish = harrisConner.mark_corner(sea1, sea1_corner)
sea2_corner = harrisConner.harris_corner(sea2, sigma)
sea2_harris_finish = harrisConner.mark_corner(sea2, sea2_corner)

if len(sea1_corner) < len(sea2_corner):
    sea_pair1 = harrisConner.find_local_match(sea1, sea2, sea1_corner, sea2_corner, 31, mode='SSD')
    sea_finish1 = harrisConner.mark_pair(sea1, sea2, sea_pair1)

    sea_pair2 = harrisConner.find_local_match(sea1, sea2, sea1_corner, sea2_corner, 31, mode='NCC')
    sea_finish2 = harrisConner.mark_pair(sea1, sea2, sea_pair2)
else:
    sea_pair1 = harrisConner.find_local_match(sea2, sea1, sea2_corner, sea1_corner, 31, mode='SSD')
    sea_finish1 = harrisConner.mark_pair(sea2, sea1, sea_pair1)

    sea_pair2 = harrisConner.find_local_match(sea2, sea1, sea2_corner, sea1_corner, 31, mode='NCC')
    sea_finish2 = harrisConner.mark_pair(sea2, sea1, sea_pair2)

cv2.imwrite('sea_compare_ssd_%s.jpg' % sigma, sea_finish1)
cv2.imwrite('sea_compare_ncc_%s.jpg' % sigma, sea_finish2)

sea_sift = siftMatch.sift_match(sea1, sea2)
cv2.imwrite('sea_sift.jpg', sea_sift)
