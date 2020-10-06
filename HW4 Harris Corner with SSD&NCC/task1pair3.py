import harrisConner
import siftMatch
import cv2

church1 = cv2.imread('church1.jpg')
church2 = cv2.imread('church2.jpg')

sigma = 2.4

church1_corner = harrisConner.harris_corner(church1, sigma)
# satellite1_harris_finish = harrisConner.mark_corner(satellite1, satellite1_corner)
# cv2.imwrite('satellite1_harrisCorner.jpg', satellite1_harris_finish)
church2_corner = harrisConner.harris_corner(church2, sigma)
if len(church1_corner) < len(church2_corner):
    church_pair1 = harrisConner.find_local_match(church1, church2, church1_corner, church2_corner, 21,
                                                   mode='SSD')
    church_finish1 = harrisConner.mark_pair(church1, church2, church_pair1)

    church_pair2 = harrisConner.find_local_match(church1, church2, church1_corner, church2_corner, 21,
                                                mode='NCC')
    church_finish2 = harrisConner.mark_pair(church1, church2, church_pair2)
else:
    church_pair1 = harrisConner.find_local_match(church2, church1, church2_corner, church1_corner, 21,
                                                   mode='SSD')
    church_finish1 = harrisConner.mark_pair(church2, church1, church_pair1)

    church_pair2 = harrisConner.find_local_match(church2, church1, church2_corner, church1_corner, 21,
                                                mode='NCC')
    church_finish2 = harrisConner.mark_pair(church2, church1, church_pair2)

cv2.imwrite('church_compare_ssd_%s.jpg' % sigma, church_finish1)
cv2.imwrite('church_compare_ncc_%s.jpg' % sigma, church_finish2)
church_sift = siftMatch.sift_match(church1, church2, mode='knn')
cv2.imwrite('church_sift.jpg', church_sift)