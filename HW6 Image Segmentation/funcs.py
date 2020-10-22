import numpy as np
import cv2
import copy


def Otsu(ch_info):
    '''

    :param img: gray-scale input image
    :param tag: True: forebackground is class C0, False:background is class C0
    :return: black-white image/channel
    '''
    img_size = len(ch_info)

    histRange = (0, 256)

    hist, bin_edge = np.histogram(ch_info, bins=256, range=histRange)
    weight_total = sum(hist * bin_edge[:-1])
    weight_back = 0
    num_back = 0

    threshold = -1
    max_btw_var = 0

    for i in range(256):
        num_back = num_back + hist[i]
        num_fore = img_size - num_back
        weight_back = weight_back + i * hist[i]  # mu of background
        weight_fore = np.int(weight_total - weight_back)  # mu of foreground
        prob_fore = num_fore / img_size

        if prob_fore == 0 or prob_fore == 1:  # foreground and background are not distinguished
            continue

        prob_back = 1 - prob_fore
        btw_var = prob_fore * prob_back * (weight_back - weight_fore) ** 2

        # the max btw_var gives out the best resolution for distinguish black and white
        if btw_var > max_btw_var:
            max_btw_var = btw_var
            threshold = i

    if threshold == -1:
        print('\nFinding threshold failed!!!!\n')

    return threshold


def segmentation_rgb(img, iteration, img_num, invert=None, tag=True):
    '''

    :param img: color-img
    :param iteration:
    :param img_num:
    :param invert: a list to choose which layer to invert, 1 means invert
    :param tag: True means foreground is C0 class
    :return:
    '''
    if invert is None:
        invert = [0, 0, 0]
    height, width = img.shape[0:2]
    out_img = np.full((height, width), 255, dtype='uint8')

    for ch in range(3):
        ch_img = img[:, :, ch]
        ch_info = copy.deepcopy(ch_img).ravel()
        for i in range(iteration):
            thresh = Otsu(ch_info)
            ch_mask = np.zeros((height, width), dtype='uint8')
            if tag is True:
                ch_mask[ch_img <= thresh] = 255
                temp_info = [j for j in ch_info if j <= thresh]
            else:
                ch_mask[ch_img > thresh] = 255
                temp_info = [j for j in ch_info if j > thresh]
            ch_info = np.asarray(temp_info)

        cv2.imwrite('img%(num)d_channel%(ch)d.jpg' % {"num": img_num, "ch": ch}, ch_mask)

        if invert[ch] == 1:
            out_img = cv2.bitwise_and(out_img, cv2.bitwise_not(ch_mask))
        else:
            out_img = cv2.bitwise_and(out_img, ch_mask)

    return out_img


def texture_segmentation(img, win_list, img_num):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    output = np.zeros((gray_img.shape[0], gray_img.shape[1], len(win_list)), dtype='uint8')

    for win_index, win_size in enumerate(win_list):
        half = int(win_size / 2)
        for i in range(half, gray_img.shape[0] - half):
            for j in range(half, gray_img.shape[1] - half):
                output[i, j, win_index] = np.var(gray_img[i - half:i + half + 1, j - half:j + half + 1])
        output[:, :, win_index] = (255 * (output[:, :, win_index] / output[:, :, win_index].max())).astype(np.uint8)

        cv2.imwrite('img%(img_num)d_texture_win%(win_size)d_texture.jpg' % {"img_num": img_num, "win_size": win_size},
                    output[:, :, win_index])
    return output


def find_contour(img):
    img_contour = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')

    # 8-neighbour detection
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] != 0:
                neighbour = img[i - 1:i + 2, j - 1:j + 2]
                if np.all(neighbour):
                    img_contour[i, j] = 0
                else:
                    img_contour[i, j] = 255

    return img_contour


def draw_contour(img, img_contour):
    img_merge = copy.deepcopy(img)
    img_merge[np.where(img_contour == 255)] = 0
    return img_merge
