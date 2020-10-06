import cv2 as cv2
import numpy as np
from scipy import signal
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def harr_filter(sigma):
    '''
    construct the harr filter matrix from lecture9 9-10(SURF)
    :param sigma: the matrix is M by M where M is the smallest even integer greater than 4*sigma
    :return: dx operator, dy operator
    '''
    filter_length = int(np.ceil(4 * sigma))
    if filter_length % 2 != 0:
        filter_length = filter_length + 1

    dx = np.ones((filter_length, filter_length))
    dx[:, :int(filter_length / 2)] = -1

    dy = np.ones((filter_length, filter_length))
    dy[int(filter_length / 2):, :] = -1

    return [dx, dy]


def harris_corner(img, sigma, percent_threshold=0.004):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''

    :param img: 
    :param sigma:
    :param percent_threshold: typically >=0.1 for the central pixel to be called a corner
    :return:
    '''
    dx, dy = harr_filter(sigma)

    Ix = signal.convolve2d(gray_img, dx, boundary='symm', mode='same')
    Iy = signal.convolve2d(gray_img, dy, boundary='symm', mode='same')

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    #  construct the window size of 5sigma*5sigma neighborhood to generate the corner matrix C
    window_size = int(5 * sigma)
    if window_size % 2 != 0:
        window_size = window_size + 1
    window = np.ones((window_size, window_size))

    sum_xx = signal.convolve2d(Ixx, window, boundary='symm', mode='same')
    sum_yy = signal.convolve2d(Iyy, window, boundary='symm', mode='same')
    sum_xy = signal.convolve2d(Ixy, window, boundary='symm', mode='same')

    tr_c = sum_xx + sum_yy
    det_c = sum_xx * sum_yy - sum_xy ** 2  # calculate the trace&det of C

    tr_c_2 = tr_c * tr_c

    ratio = det_c / tr_c_2
    ratio[np.isnan(ratio)] = 0
    ratio_flat = ratio.copy()
    ratio_flat = np.ravel(ratio_flat)
    num_above_threshold = int(len(ratio_flat) * percent_threshold)
    ratio_flat[::-1].sort()
    threshold = ratio_flat[num_above_threshold - 1]

    i, j = np.where(ratio >= threshold)  #
    pixel_list = zip(i, j)

    return list(pixel_list)


def mark_corner(img, corner_list, radius=4, color=(0, 255, 255), thickness=1):
    '''
    mark coner pixels with circle
    :param img: input image
    :param corner_list:
    :param radius:
    :param color: set #FFFF99 as default
    :return:img with circles
    '''
    for i, j in corner_list:
        cv2.circle(img, (j, i), radius, color, thickness, lineType=8, shift=0)
    return img


def find_coor(img1, img2, corner_point1, corner_point2, window_size, mode=None):
    if mode == None:
        print('Please choose mode (SSD or NCC)\n')
        return

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    half_window = int(window_size / 2)

    '''
    Set boundaries for img1&img2 if the window is not in the image,
    cut the window according to the image boundary
    for image1:
    '''
    x_min_flag = 0
    x_max_flag = 0
    y_min_flag = 0
    y_max_flag = 0
    x1_min, x1_max = corner_point1[0] - half_window, corner_point1[0] + half_window
    if x1_min < 0:
        x1_min = 0
        x_min_flag = 1
    if x1_max > img1.shape[0]:
        x1_max = img1.shape[0] - 1
        x_max_flag = 1

    y1_min, y1_max = corner_point1[1] - half_window, corner_point1[1] + half_window
    if y1_min < 0:
        y1_min = 0
        y_min_flag = 1
    if y1_max > img1.shape[1]:
        y1_max = img1.shape[1] - 1
        y_max_flag = 1

    x2_min, x2_max = corner_point2[0] - half_window, corner_point2[0] + half_window
    if x2_min < 0:
        x2_min = 0
        x_min_flag = 1
    if x2_max > img2.shape[0]:
        x2_max = img2.shape[0] - 1
        x_max_flag = 1

    y2_min, y2_max = corner_point2[1] - half_window, corner_point2[1] + half_window
    if y2_min < 0:
        y2_min = 0
        y_min_flag = 1
    if y2_max > img2.shape[1]:
        y2_max = img2.shape[1] - 1
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

    neighborhood1 = gray_img1[y1_min:y1_max， x1_min:x1_max]
    neighborhood2 = gray_img2[y2_min:y2_max， x2_min:x2_max]

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


def find_local_match(img1, img2, corners1, corners2, window_size, mode=None):
    if mode == None:
        print('Please select a mode(SSD or NCC):\n')

    corners1 = np.array(corners1)
    corners2 = np.array(corners2)

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
                temp = np.where(ssd == min(ssd))[0]
                match_pair.append([point1, corners2[temp][0]])

    elif mode == 'NCC':
        threshold = 0.7
        # matched_pair is to save the points got paired from corresponding points in img2
        match_pair = list()
        for point1 in corners1:
            ncc = list()
            for point2 in corners2:
                ncc.append(find_coor(img1, img2, point1, point2, window_size, mode='NCC'))
            if max(ncc) > threshold:
                temp = np.where(ncc == max(ncc))[0]
                match_pair.append([point1, corners2[temp][0]])

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
        cv2.circle(img, (point1[1], point1[0]), radius, color, thickness, lineType=8, shift=0)
        cv2.circle(img, (point2[1] + int(width / 2 - 1), point2[0]), radius, color, thickness, lineType=8, shift=0)
        cv2.line(img, (point1[1], point1[0]), (point2[1] + int(width / 2 - 1), point2[0]), (255, 255, 0), thickness)

    return img
