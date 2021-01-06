import cv2
import numpy as np


# reference: https://stackoverflow.com/questions/38265364/census-transform-in-python-opencv
def census_trans(img, window_size=3):
    """
    :param img: gray-scale
    :param window_size: int odd-value
    :return: census transform of img
    """
    half_window_size = window_size // 2

    img = cv2.copyMakeBorder(img, top=half_window_size, left=half_window_size, right=half_window_size,
                             bottom=half_window_size, borderType=cv2.BORDER_CONSTANT, value=0)
    height, width = img.shape[:2]
    census = np.zeros((height - half_window_size * 2, width - half_window_size * 2), dtype=np.uint8)
    center_pixels = img[half_window_size:height - half_window_size, half_window_size:width - half_window_size]

    offsets = [(row, col) for row in range(half_window_size) for col in range(half_window_size) if
               not row == half_window_size + 1 == col]
    for (row, col) in offsets:
        census = (census << 1) | (img[row:row + height - half_window_size * 2,
                                  col:col + width - half_window_size * 2] >= center_pixels)
    return census


def column_cost(left_col, right_col):
    # Column-wise Hamming edit distance
    return np.sum(np.unpackbits(np.bitwise_xor(left_col, right_col), axis=1), axis=1).reshape(left_col.shape[0],
                                                                                              left_col.shape[1])


def cost(left, right, window_size=3, disparity=0):
    """
    Compute cost difference between left and right grayscale imgs.
    :param left :img
    :param right: img
    :param window_size: int odd-valued
    :param disparity: int
    :return:
    """
    ct_left = census_trans(left, window_size=window_size)
    ct_right = census_trans(right, window_size=window_size)
    height, width = ct_left.shape
    C = np.full(shape=(height, width), fill_value=0)
    for col in range(disparity, width):
        C[:, col] = column_cost(
            ct_left[:, col:col + 1],
            ct_right[:, col - disparity:col - disparity + 1]
        ).reshape(ct_left.shape[0])
    return C


def min_cost(ct_list):
    # ct_list = np.array(ct_list)
    # min_ct = np.zeros(ct_list[0].shape)
    # for i in range(ct_list[0].shape[0]):
    #     for j in range(ct_list[0].shape[1]):
    #         min_ct[i, j] = min(ct_list[:, i, j])
    total_cost = [np.sum(ct) for ct in ct_list]
    total_cost = np.array(total_cost)
    idx = np.where(total_cost == min(total_cost))[0]
    idx = idx[0]
    min_ct = ct_list[idx]
    return min_ct


def norm(img):
    return cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)


def find_valid(truth_map, est_map, delta=1):
    abs_diff = np.abs(truth_map - est_map)
    valid_pixel = np.count_nonzero(abs_diff < delta)
    out = np.where(abs_diff <= delta, 255, 0)
    return valid_pixel, out


if __name__ == '__main__':

    left = cv2.imread('Task2_Images\Left.ppm', 0)
    right = cv2.imread('Task2_Images\Right.ppm', 0)
    truth = cv2.imread('Task2_Images\left_truedisp.pgm', 0)
    truth = truth.astype(np.float32)
    truth /= 16
    truth = truth.astype(np.int16)
    truth_mask = cv2.imread('Task2_Images\mask0nocc.png', 0)
    # cv2.imshow('truth mask', truth_mask)
    # cv2.waitKey(0)
    d_max = np.max(truth) + 1

    window_size = 5
    print('Window size=', window_size)
    ct_left = census_trans(left, window_size)
    norm_ct_left = norm(ct_left)
    ct_right = census_trans(right, window_size)
    norm_ct_right = norm(ct_right)

    ct_costs = []
    for disparity in range(d_max):
        # d = [0,...,d_max]
        temp_cost = cost(left, right, window_size, disparity)
        ct_costs.append(temp_cost)

    census_combine = np.vstack([np.hstack([left, right]), np.hstack([norm_ct_left, norm_ct_right])])
    cv2.imwrite('census_win%d.jpg' % window_size, norm(census_combine))
    cv2.imshow('left/right grayscale/census', census_combine)
    cv2.waitKey(0)

    min_map = min_cost(ct_costs)
    cv2.imshow('min_cost map', norm(min_map))
    cv2.waitKey(0)
    cv2.imwrite('min_cost_map_win_%d.jpg' % window_size, min_map)

    total_pixel = np.count_nonzero(truth_mask == 255)

    # delta=1
    valid_num, out = find_valid(truth, min_map, delta=1)
    acc1 = valid_num / total_pixel
    print('delta=1, acc: ', acc1)
    # for i in range(len(out_list)):
    cv2.imwrite('mask_win_%ddelta_%d.jpg' % (window_size, 1), out)
    cv2.imshow('mask with delta=1', norm(out))
    cv2.waitKey(0)

    # delta=2
    valid_num, out = find_valid(truth, min_map, delta=2)
    acc2 = valid_num / total_pixel
    print('delta=2, acc: ', acc2)
    cv2.imwrite('mask_win_%ddelta_%d.jpg' % (window_size, 2), out)
    cv2.imshow('mask with delta=2', norm(out))
    cv2.waitKey(0)

    # delta=4
    valid_num, out = find_valid(truth, min_map, delta=4)
    acc3 = valid_num / total_pixel
    print('delta=4, acc: ', acc3)
    # cv2.imwrite('mask_win_%ddelta_%d.jpg' % (window_size, 4), out)
    cv2.imshow('mask with delta=4', norm(out))
    cv2.waitKey(0)