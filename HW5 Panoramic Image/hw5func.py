import numpy as np
import cv2
import scipy.optimize
import math
import random


def sift_features(img):
    '''

    :param img: original colored picture
    :return: colored picture with interest points & keypoint & des
    '''

    sift = cv2.SIFT_create(nfeatures=5000, nOctaveLayers=4, contrastThreshold=0.03, edgeThreshold=10, sigma=4)
    kp, des = sift.detectAndCompute(img, None)  # kp will be a list of keypoints and des is a numpy array of shape

    corners_list = cv2.KeyPoint_convert(kp)
    corners_list = np.array(corners_list)

    return corners_list


def find_coor(img1, img2, corner_point1, corner_point2, window_size, mode=None):
    if mode is None:
        print('Please choose mode (SSD or NCC)\n')
        return

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    half_window = int(window_size / 2)

    # Set boundaries for img1&img2 if the window is not in the image,
    # cut the window according to the image boundary
    # for image1:

    x_min_flag = 0
    x_max_flag = 0
    y_min_flag = 0
    y_max_flag = 0
    x1_min, x1_max = corner_point1[0] - half_window, corner_point1[0] + half_window
    if x1_min < 0:
        x1_min = 0
        x_min_flag = 1
    if x1_max >= img1.shape[1]:
        x1_max = img1.shape[1] - 1
        x_max_flag = 1

    y1_min, y1_max = corner_point1[1] - half_window, corner_point1[1] + half_window
    if y1_min < 0:
        y1_min = 0
        y_min_flag = 1
    if y1_max >= img1.shape[0]:
        y1_max = img1.shape[0] - 1
        y_max_flag = 1

    x2_min, x2_max = corner_point2[0] - half_window, corner_point2[0] + half_window
    if x2_min < 0:
        x2_min = 0
        x_min_flag = 1
    if x2_max >= img2.shape[1]:
        x2_max = img2.shape[1] - 1
        x_max_flag = 1

    y2_min, y2_max = corner_point2[1] - half_window, corner_point2[1] + half_window
    if y2_min < 0:
        y2_min = 0
        y_min_flag = 1
    if y2_max >= img2.shape[0]:
        y2_max = img2.shape[0] - 1
        y_max_flag = 1

    if x_min_flag:
        x_min_length = min(corner_point1[0] - x1_min, corner_point2[0] - x2_min)
        x1_min = corner_point1[0] - x_min_length
        x2_min = corner_point2[0] - x_min_length
    if y_min_flag:
        y_min_length = min(corner_point1[1] - y1_min, corner_point2[1] - y2_min)
        y1_min = corner_point1[1] - y_min_length
        y2_min = corner_point2[1] - y_min_length
    if x_max_flag:
        x_max_length = min(x1_max - corner_point1[0], x2_max - corner_point2[0])
        x1_max = corner_point1[0] + x_max_length
        x2_max = corner_point2[0] + x_max_length
    if y_max_flag:
        y_max_length = min(y1_max - corner_point1[1], y2_max - corner_point2[1])
        y1_max = corner_point1[1] + y_max_length
        y2_max = corner_point2[1] + y_max_length

    neighborhood1 = gray_img1[y1_min:y1_max, x1_min:x1_max]
    neighborhood2 = gray_img2[y2_min:y2_max, x2_min:x2_max]

    if mode == 'SSD':
        ssd = np.sum((neighborhood1 - neighborhood2) ** 2)
        return ssd

    if mode == 'NCC':
        m1 = np.mean(neighborhood1)
        m2 = np.mean(neighborhood2)

        a = neighborhood1 - m1
        b = neighborhood2 - m2
        numerator = np.sum(a * b)
        denominator = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))

        ncc = numerator / denominator
        return ncc


def find_local_match(img1, img2, window_size, mode=None):
    if mode is None:
        print('Please select a mode(SSD or NCC):\n')
        return 0

    corners1 = sift_features(img1).astype(int)
    corners2 = sift_features(img2).astype(int)

    if mode == 'SSD':
        threshold = 10000
        # matched_pair is to save the points got paired from corresponding points in img2
        match_pair = list()
        for point1 in corners1:
            ssd = list()
            for point2 in corners2:
                ssd.append(find_coor(img1, img2, point1, point2, window_size, mode='SSD'))
            if min(ssd) < threshold:
                ssd = np.array(ssd)
                temp = np.where(ssd == min(ssd))[0][0]
                match_pair.append([point1, corners2[temp]])

    elif mode == 'NCC':
        threshold = 0.8
        # matched_pair is to save the points got paired from corresponding points in img2
        match_pair = list()
        for point1 in corners1:
            ncc = list()
            for point2 in corners2:
                ncc.append(find_coor(img1, img2, point1, point2, window_size, mode='NCC'))
            if max(ncc) > threshold:
                temp = np.where(ncc == max(ncc))[0][0]
                match_pair.append([point1, corners2[temp]])

    return match_pair


def mark_pair(img1, img2, match_pair, radius=4, color=(0, 255, 255), thickness=1):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    width = w1 + w2
    # create empty matrix
    img = np.zeros((max(h1, h2), width, 3), np.uint8)

    # combine 2 images
    img[:h1, :w1, :3] = img1
    img[:h2, w1:w1 + w2, :3] = img2

    for point1, point2 in match_pair:
        cv2.circle(img, (point1[0], point1[1]), radius, color, thickness, lineType=8, shift=0)
        cv2.circle(img, (point2[0] + w1, point2[1]), radius, color, thickness, lineType=8, shift=0)
        cv2.line(img, (point1[0], point1[1]), (point2[0] + w1, point2[1]), (255, 255, 0), thickness)

    return img


def homography_estimate(src_pts, dst_pts):
    h = np.zeros((3, 3))

    A = np.zeros((2 * len(src_pts), 8))  # initialize the A matrix
    b = np.zeros((2 * len(src_pts), 1))
    for i in range(len(src_pts)):
        A[2 * i] = [0, 0, 0, -src_pts[i][0], -src_pts[i][1], -1, dst_pts[i][1] * src_pts[i][0],
                    dst_pts[i][1] * src_pts[i][1]]
        A[2 * i + 1] = [src_pts[i][0], src_pts[i][1], 1, 0, 0, 0, -dst_pts[i][0] * src_pts[i][0],
                        -dst_pts[i][0] * src_pts[i][1]]
        b[2 * i] = -dst_pts[i][1]
        b[2 * i + 1] = dst_pts[i][0]

    ATA = np.matmul(A.transpose(), A)
    ATA_inv = np.linalg.inv(ATA)
    pseudo_A_inv = np.matmul(ATA_inv, A.transpose())
    h = np.matmul(pseudo_A_inv, b)

    temp = np.array([[1]])
    h = np.r_[h, temp]
    h = h.reshape((3, 3))

    return h


def RANSAC(match_pair, delta, n, p, epsilon):
    '''


    :param delta: decision threshold to construct the inlier set
    :param n: prob(at least 1 of the N trials will be free of the outliers
    :param p:
    :param epsilon: prob( a random chosen point pair is an outlier)
    :return:
    '''
    matched_img1 = np.array(match_pair)[:, 0]
    matched_img2 = np.array(match_pair)[:, 1]

    N = int(math.log(1 - p) / math.log(1 - (1 - epsilon) ** n))  # number of trials
    n_total = len(matched_img1)  # total number of correspondence
    M = int((1 - epsilon) * n_total)  # minimum value for the size of the inlier set

    num_inliers = -1
    h_final = np.zeros((3, 3))  # initialize the homography matrix h

    for trial in range(N):
        # Randomly select 1 pair from matched pairs
        random_index = random.sample(range(0, n_total), n)
        src_pts = np.array([matched_img1[i] for i in random_index])
        dst_pts = np.array([matched_img2[i] for i in random_index])
        h_temp = homography_estimate(src_pts, dst_pts)

        temp_inliers_loc, num_temp_inliers, temp_outliers_loc = find_inliers(matched_img1, matched_img2, h_temp, delta)
        if num_temp_inliers > num_inliers:
            num_inliers = num_temp_inliers
            inlier_loc = temp_inliers_loc
            outlier_loc = temp_outliers_loc
            h_final = h_temp

    if num_inliers > M:
        return h_final, inlier_loc, outlier_loc
    else:
        print("Finding inliers failed!\n")
        return 0, 0, 0


def mark_inlier_outlier(img1, img2, match_pair, inlier_loc, outlier_loc, radius=4, color=(0, 255, 255), thickness=1):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    width = w1 + w2
    # create empty matrix
    img = np.zeros((max(h1, h2), width, 3), np.uint8)

    # combine 2 images
    img[:h1, :w1, :3] = img1
    img[:h2, w1:w1 + w2, :3] = img2

    inlier_img1, inlier_img2 = get_in_out_liers_pts(inlier_loc, match_pair)
    for i in range(inlier_img1.shape[0]):
        point1 = inlier_img1[i]
        point2 = inlier_img2[i]
        cv2.circle(img, (point1[0], point1[1]), radius, color, thickness, lineType=8, shift=0)
        cv2.circle(img, (point2[0] + w1, point2[1]), radius, color, thickness, lineType=8, shift=0)
        cv2.line(img, (point1[0], point1[1]), (point2[0] + w1, point2[1]), (255, 255, 0), thickness)

    outlier_img1, outlier_img2 = get_in_out_liers_pts(outlier_loc, match_pair)
    for i in range(outlier_img1.shape[0]):
        point1 = outlier_img1[i]
        point2 = outlier_img2[i]
        cv2.circle(img, (point1[0], point1[1]), radius, (0, 0, 255), thickness, lineType=8, shift=0)
        cv2.circle(img, (point2[0] + w1, point2[1]), radius, (0, 0, 255), thickness, lineType=8, shift=0)
        cv2.line(img, (point1[0], point1[1]), (point2[0] + w1, point2[1]), (255, 51, 153), thickness)

    return img


def find_inliers(src_pts, dst_pts, h_temp, delta):
    predict_pts = np.zeros(dst_pts.shape)
    for i in range(src_pts.shape[0]):
        src_pt = np.array([src_pts[i][0], src_pts[i][1], 1]).transpose()
        point_temp = np.matmul(h_temp, src_pt)
        predict_pts[i][0] = point_temp[0] / point_temp[2]
        predict_pts[i][1] = point_temp[1] / point_temp[2]

    error = np.square(predict_pts - dst_pts)
    dist = np.sqrt(error[:, 0] + error[:, 1])

    inliers_loc = np.where(dist <= delta)
    outliers_loc = np.where(dist > delta)
    num_inliers = len(inliers_loc[0])
    return inliers_loc, num_inliers, outliers_loc


def get_in_out_liers_pts(inliers_loc, match_pair):  # get_outliers_pts is the same, except the name
    matched_img1 = np.array(match_pair)[:, 0]
    matched_img2 = np.array(match_pair)[:, 1]

    inliers_img1 = [matched_img1[i] for i in inliers_loc]
    inliers_img1 = np.array(inliers_img1).squeeze(axis=0)
    inliers_img2 = [matched_img2[i] for i in inliers_loc]
    inliers_img2 = np.array(inliers_img2).squeeze(axis=0)

    return inliers_img1, inliers_img2


def loss_Func(h, match_pair):
    h_trans = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]])
    matched_img1 = np.array(match_pair)[:, 0:2]
    matched_img2 = np.array(match_pair)[:, 2:4]
    homo_matched_img1 = np.ones((match_pair.shape[0], 3))
    homo_matched_img1[:, 0:2] = matched_img1
    homo_matched_img1_t = homo_matched_img1.transpose()

    homo_predict = np.matmul(h_trans, homo_matched_img1_t)  # return all the predicted pts
    predict = np.transpose((homo_predict[0:2, :] / homo_predict[2, :]))

    error = predict - matched_img2
    return error.flatten()


def nonlinear_LM(h_ransac, inliers_loc, match_pair):
    inliers_img1, inliers_img2 = get_in_out_liers_pts(inliers_loc, match_pair)
    inliers_pairs = np.zeros((inliers_img1.shape[0], 4))
    inliers_pairs[:, 0:2] = inliers_img1
    inliers_pairs[:, 2:4] = inliers_img2
    h_init = [h_ransac[0][0], h_ransac[0][1], h_ransac[0][2], h_ransac[1][0],
              h_ransac[1][1], h_ransac[1][2], h_ransac[2][0], h_ransac[2][1]]
    sol = scipy.optimize.least_squares(loss_Func, h_init, method='lm', args=[inliers_pairs])
    h_refined = np.array([[sol.x[0], sol.x[1], sol.x[2]],
                          [sol.x[3], sol.x[4], sol.x[5]],
                          [sol.x[6], sol.x[7], 1]])
    return h_refined


def find_new_roi_set(img, h):
    height, width = img.shape[0:2]
    # Find ROI pqrs in the frame corners
    p = [0, 0, 1]
    q = [0, height, 1]
    r = [width, height, 1]
    s = [width, 0, 1]
    roi_set = np.concatenate((p, q, r, s), axis=0)
    roi_set = np.transpose(roi_set.reshape((4, 3)))

    new_roi_set = np.matmul(h, roi_set)
    for i in range(4):
        new_roi_set[:, i] = new_roi_set[:, i] / new_roi_set[2][i]

    new_roi_set = new_roi_set.astype(int)

    return new_roi_set


def panaromic_img(roi_13, roi_23, roi_33, roi_43, roi_53):
    final_roi = np.concatenate((roi_13, roi_23,
                                roi_33, roi_43, roi_53), axis=1)

    height_min = min(final_roi[1])
    height_max = max(final_roi[1])
    width_min = min(final_roi[0])
    width_max = max(final_roi[0])

    final_height = height_max - height_min + 1
    final_width = width_max - width_min + 1
    # create the initial image for the next mapping
    final_img = np.zeros((final_height, final_width, 3), np.uint8)

    return final_img, height_min, width_min


def mapping(h, input_image, output_image,
            height_min, width_min):
    h_inv = np.linalg.inv(h)

    input_height, input_width = input_image.shape[0:2]
    output_height, output_width = output_image.shape[0:2]

    for j in range(output_height):
        for i in range(output_width):
            temp_point = np.matmul(h_inv, np.array([i + width_min, j + height_min, 1]))
            point = temp_point / temp_point[2]
            if 0 < point[0] < input_width and 0 < point[1] < input_height:
                output_image[j][i] = input_image[int(point[1])][int(point[0])]

    return output_image
