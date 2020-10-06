import cv2 as cv2
import harrisConner
import siftMatch

building1 = cv2.imread('building1.jpg')
building2 = cv2.imread('building2.jpg')

sigma = 3

building1_corner = harrisConner.harris_corner(building1, sigma)
building2_corner = harrisConner.harris_corner(building2, sigma)

if len(building1_corner) < len(building2_corner):
    building_pair1 = harrisConner.find_local_match(building1, building2, building1_corner, building2_corner, 21,
                                                   mode='SSD')
    building_finish1 = harrisConner.mark_pair(building1, building2, building_pair1)

    building_pair2 = harrisConner.find_local_match(building1, building2, building1_corner, building2_corner, 21,
                                                   mode='NCC')
    building_finish2 = harrisConner.mark_pair(building1, building2, building_pair2)
else:
    building_pair1 = harrisConner.find_local_match(building2, building1, building2_corner, building1_corner, 21,
                                                   mode='SSD')
    building_finish1 = harrisConner.mark_pair(building2, building1, building_pair1)

    building_pair2 = harrisConner.find_local_match(building2, building1, building2_corner, building1_corner, 21,
                                                   mode='NCC')
    building_finish2 = harrisConner.mark_pair(building2, building1, building_pair2)

cv2.imwrite('building_compare_ssd_%s.jpg' % sigma, building_finish1)
cv2.imwrite('building_compare_ncc_%s.jpg' % sigma, building_finish2)

building_sift = siftMatch.sift_match(building1, building2, max_show=100)
cv2.imwrite('building_sift.jpg', building_sift)
