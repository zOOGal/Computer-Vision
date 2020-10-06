import harrisConner
import siftMatch
import cv2

satellite1 = cv2.imread('satellite1.jpg')
satellite2 = cv2.imread('satellite2.jpg')

sigma = 3

satellite1_corner = harrisConner.harris_corner(satellite1, sigma)
satellite2_corner = harrisConner.harris_corner(satellite2, sigma)
if len(satellite1_corner) < len(satellite2_corner):
    satellite_pair1 = harrisConner.find_local_match(satellite1, satellite2, satellite1_corner, satellite2_corner, 21,
                                                   mode='SSD')
    satellite_finish1 = harrisConner.mark_pair(satellite1, satellite2, satellite_pair1)

    satellite_pair2 = harrisConner.find_local_match(satellite1, satellite2, satellite1_corner, satellite2_corner, 21,
                                                   mode='NCC')
    satellite_finish2 = harrisConner.mark_pair(satellite1, satellite2, satellite_pair2)
else:
    satellite_pair1 = harrisConner.find_local_match(satellite2, satellite1, satellite2_corner, satellite1_corner, 21,
                                                   mode='SSD')
    satellite_finish1 = harrisConner.mark_pair(satellite2, satellite1, satellite_pair1)

    satellite_pair2 = harrisConner.find_local_match(satellite2, satellite1, satellite2_corner, satellite1_corner, 21,
                                                   mode='NCC')
    satellite_finish2 = harrisConner.mark_pair(satellite2, satellite1, satellite_pair2)

cv2.imwrite('satellite_compare_ssd%s.jpg' % sigma, satellite_finish1)
cv2.imwrite('satellite_compare_ncc%s.jpg' % sigma, satellite_finish2)
satellite_sift = siftMatch.sift_match(satellite1, satellite2)
cv2.imwrite('satellite_sift.jpg', satellite_sift)
