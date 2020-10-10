import hw5func
import cv2
import numpy as np

sigma = 2
delta = 3 * sigma
n = 6
p = 0.9
epsilon = 0.5

img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')
img3 = cv2.imread('img3.jpg')
img4 = cv2.imread('img4.jpg')
img5 = cv2.imread('img5.jpg')

pair1 = hw5func.find_local_match(img1, img2, 25, 'NCC')
pair1_img = hw5func.mark_pair(img1, img2, pair1)
h_12, inliers12, outliers12 = hw5func.RANSAC(pair1, delta, n, p, epsilon)
pair1_in_out = hw5func.mark_inlier_outlier(img1, img2, pair1, inliers12, outliers12)
h_12_refined = hw5func.nonlinear_LM(h_12, inliers12, pair1)
cv2.imwrite('pair1.jpg', pair1_img)
cv2.imwrite('pair1with_in&out.jpg', pair1_in_out)

pair2 = hw5func.find_local_match(img2, img3, 25, 'NCC')
pair2_img = hw5func.mark_pair(img2, img3, pair2)
h_23, inliers23, outliers23 = hw5func.RANSAC(pair2, delta, n, p, epsilon)
pair2_in_out = hw5func.mark_inlier_outlier(img2, img3, pair2, inliers23, outliers23)
h_23_refined = hw5func.nonlinear_LM(h_23, inliers23, pair2)
cv2.imwrite('pair2.jpg', pair2_img)
cv2.imwrite('pair2with_in&out.jpg', pair2_in_out)

pair3 = hw5func.find_local_match(img3, img4, 25, 'NCC')
pair3_img = hw5func.mark_pair(img3, img4, pair3)
h_34, inliers34, outliers34 = hw5func.RANSAC(pair3, delta, n, p, epsilon)
pair3_in_out = hw5func.mark_inlier_outlier(img3, img4, pair3, inliers34, outliers34)
h_34_refined = hw5func.nonlinear_LM(h_34, inliers34, pair3)
cv2.imwrite('pair3.jpg', pair3_img)
cv2.imwrite('pair3with_in&out.jpg', pair3_in_out)

pair4 = hw5func.find_local_match(img4, img5, 25, 'NCC')
pair4_img = hw5func.mark_pair(img4, img5, pair4)
h_45, inliers45, outliers45 = hw5func.RANSAC(pair4, delta, n, p, epsilon)
pair4_in_out = hw5func.mark_inlier_outlier(img4, img5, pair4, inliers45, outliers45)
h_45_refined = hw5func.nonlinear_LM(h_45, inliers45, pair4)
cv2.imwrite('pair4.jpg', pair4_img)
cv2.imwrite('pair4with_in&out.jpg', pair4_in_out)

h13 = np.matmul(h_12, h_23)
h13 = h13 / h13[2][2]
h13_refined = np.matmul(h_12_refined, h_23_refined)
h13_refined = h13_refined / h13_refined[2][2]

h33 = np.diag(np.ones(3))

h43 = np.linalg.inv(h_34)
h43_refined = np.linalg.inv(h_34_refined)

h_35 = np.matmul(h_34, h_45)
h_35 = h_35 / h_35[2][2]
h_53 = np.linalg.inv(h_35)
h_35_refined = np.matmul(h_34_refined, h_45_refined)
h_35_refined = h_35_refined / h_35_refined[2][2]
h_53_refined = np.linalg.inv(h_35_refined)

final_img, height_min, width_min = hw5func.panaromic_img(hw5func.find_new_roi_set(img1, h13),
                                                         hw5func.find_new_roi_set(img2, h_23),
                                                         hw5func.find_new_roi_set(img3, h33),
                                                         hw5func.find_new_roi_set(img4, h43),
                                                         hw5func.find_new_roi_set(img5, h_53))
final_img = hw5func.mapping(h13, img1, final_img, height_min, width_min)
final_img = hw5func.mapping(h_23, img2, final_img, height_min, width_min)
final_img = hw5func.mapping(h33, img3, final_img, height_min, width_min)
final_img = hw5func.mapping(h43, img4, final_img, height_min, width_min)
final_img = hw5func.mapping(h_53, img5, final_img, height_min, width_min)
cv2.imwrite('final_RANSAC.jpg', final_img)


final_img_refined, height_min_refined, width_min_refined = hw5func.panaromic_img(
    hw5func.find_new_roi_set(img1, h13_refined),
    hw5func.find_new_roi_set(img2, h_23_refined),
    hw5func.find_new_roi_set(img3, h33),
    hw5func.find_new_roi_set(img4, h43_refined),
    hw5func.find_new_roi_set(img5, h_53_refined))
final_img_refined = hw5func.mapping(h13_refined, img1, final_img_refined, height_min_refined, width_min_refined)
final_img_refined = hw5func.mapping(h_23_refined, img2, final_img_refined, height_min_refined, width_min_refined)
final_img_refined = hw5func.mapping(h33, img3, final_img_refined, height_min_refined, width_min_refined)
final_img_refined = hw5func.mapping(h43_refined, img4, final_img_refined, height_min_refined, width_min_refined)
final_img_refined = hw5func.mapping(h_53_refined, img5, final_img_refined, height_min_refined, width_min_refined)
cv2.imwrite('refined_RANSAC.jpg', final_img_refined)
