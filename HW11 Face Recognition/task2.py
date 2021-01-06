import numpy as np
import os
import cv2
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
import pdb


def load_img(img_path):
    img_name_list = os.listdir(img_path)
    img_list = list()
    for img_name in img_name_list:
        img = cv2.imread(os.path.join(img_path, img_name), 0)
        img_list.append(img)
    return img_list


def harr_kernel(img):
    horizontal_kernel_list = list()
    vertical_kernel_list = list()
    unit_kernel_height = 1
    unit_kernel_width = 2
    for h in range(1, img.shape[0], unit_kernel_height):
        white = np.zeros((h, 2))
        black = np.ones((h, 2))
        kernel = np.vstack([white, black])
        vertical_kernel_list.append(kernel)
    for w in range(1, img.shape[1], unit_kernel_width):
        white = np.zeros((1, w))
        black = np.ones((1, w))
        kernel = np.hstack([white, black])
        horizontal_kernel_list.append(kernel)
    return horizontal_kernel_list, vertical_kernel_list


def compute_integral(img):
    int_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            int_img[i, j] = np.sum(img[:i + 1, :j + 1])
    return int_img


def sum_rec(int_img, corner_list):
    """
    the sum of values​in a region is only related to the integral graph of the four vertices in this region
     (the upper-left, upper-right, down-left, down-right corners)
    :param int_img: integral img of the whole img
    :param corner_list: positions of this four corners
    :return:
    """
    up_left = int_img[int(corner_list[0, 0]), int(corner_list[0, 1])]
    up_right = int_img[int(corner_list[1, 0]), int(corner_list[1, 1])]
    dn_left = int_img[int(corner_list[2, 0]), int(corner_list[2, 1])]
    dn_right = int_img[int(corner_list[3, 0]), int(corner_list[3, 1])]
    total = dn_right - dn_left - up_right + up_left
    return total


def generate_harr_feature(int_img):
    horizontal_kernel_list, vertical_kernel_list = harr_kernel(int_img)

    feature_list = list()

    # horizontal direction
    for kernel in horizontal_kernel_list:
        kernel_height, kernel_width = kernel.shape
        for row in range(int_img.shape[0] - kernel_height):
            for col in range(int_img.shape[1] - kernel_width):
                white_corner_list = np.array([[row, col], [row, col + int(kernel_width / 2)],
                                              [row + kernel_height, col],
                                              [row + kernel_height, col + int(kernel_width / 2)]])
                black_corner_list = np.array([[row, col + int(kernel_width / 2)], [row, col + kernel_width],
                                              [row + kernel_height, col + int(kernel_width / 2)],
                                              [row + kernel_height, col + kernel_width]])
                # feature = sum(black) - sum(white)
                feature = sum_rec(int_img, black_corner_list) - sum_rec(int_img, white_corner_list)
                feature_list.append(feature)

    # vertical direction
    for kernel in vertical_kernel_list:
        kernel_height, kernel_width = kernel.shape
        for row in range(int_img.shape[0] - kernel_height):
            for col in range(int_img.shape[1] - kernel_width):
                white_corner_list = np.array([[row, col], [row, col + kernel_width],
                                              [row + int(kernel_height / 2), col],
                                              [row + int(kernel_height / 2), col + kernel_width]])
                black_corner_list = np.array([[row, col + kernel_width],
                                              [row + int(kernel_height / 2), col + kernel_width],
                                              [row + kernel_height, col],
                                              [row + kernel_height, col + kernel_width]])
                # feature = sum(black) - sum(white)
                feature = sum_rec(int_img, black_corner_list) - sum_rec(int_img, white_corner_list)
                feature_list.append(feature)
    return np.array(feature_list)


def best_weak_classifier(feature_list, target, weight, n_positive):
    n_features, n_imgs = feature_list.shape
    best_min_error = np.inf
    t_positive = np.sum(weight[:, :n_positive])
    t_negative = np.sum(weight[:, :n_positive:])
    # best_feature = None
    best_polar = 0
    best_thresh = np.inf
    best_clf_output = None
    best_x = 0
    for i in range(n_features):
        temp_feature = feature_list[i, :]
        sorted_temp_feature = np.sort(temp_feature)
        idx_sorted_temp_feature = np.argsort(temp_feature)
        sorted_target = np.squeeze(target, axis=0)[idx_sorted_temp_feature]
        sorted_weight = weight[:, idx_sorted_temp_feature]

        s_positive = np.cumsum(sorted_weight * sorted_target)
        s_negative = np.cumsum(sorted_weight) - s_positive

        err1 = s_positive + (t_negative - s_negative)
        err2 = s_negative + (t_positive - s_positive)

        min_error_list = np.minimum(err1, err2)
        min_error = np.amin(min_error_list)
        thresh_idx = np.argmin(min_error_list)

        clf_output = np.zeros((n_imgs, 1))

        if err1[thresh_idx] > err2[thresh_idx]:
            polar = 1
            clf_output[thresh_idx:,:] = 1
        else:
            polar = -1
            clf_output[:thresh_idx, :] = 1
        clf_output[idx_sorted_temp_feature] = clf_output
        if min_error < best_min_error:
            best_min_error = min_error
            best_clf_output = clf_output
            best_x = i
            # best weak classifier
            # ht(x) = h(x, ft, pt, θt) for current iteration t. ft is the feature, pt is the polarity,
            # and θt is the threshold value that minimize the error
            # best_feature = temp_feature
            best_polar = polar
            if thresh_idx == 0:
                best_thresh = sorted_temp_feature[thresh_idx] - 0.1
            elif thresh_idx == sorted_temp_feature.shape[0]:
                best_thresh = sorted_temp_feature[thresh_idx] + 0.1
            else:
                best_thresh = (sorted_temp_feature[thresh_idx - 1] + sorted_temp_feature[thresh_idx]) / 2

    ht_parameter = [best_x, best_polar, best_thresh, best_min_error]
    return ht_parameter, best_clf_output


def cascade_classifiers(feature_list, pos_target, neg_target, tolerant_tpr, tolerant_fpr, n_iteration=20):
    n_pos = pos_target.shape[1]
    n_neg = neg_target.shape[1]

    # initialize the weights
    positive_weights = np.ones(pos_target.shape) / (2 * n_pos)
    negative_weights = np.ones(neg_target.shape) / (2 * n_neg)
    weights = np.hstack((positive_weights, negative_weights))
    targets = np.hstack((pos_target, neg_target))
    alpha = list()
    ht = list()
    pred_list = list()

    for itr in range(n_iteration):
        # normalize weights
        norm_weight = weights / np.sum(weights)
        temp_parameter, temp_output = best_weak_classifier(feature_list, targets, norm_weight, n_pos)
        pred_list.append(np.transpose(temp_output))
        ht.append(temp_parameter)
        epsilon = temp_parameter[3]
        temp_alpha = (1 - epsilon) / epsilon
        alpha.append(np.log(temp_alpha))
        weights = weights * (temp_alpha ** (1 - np.logical_xor(targets, np.transpose(temp_output))))

        temp_strong = np.matmul(np.transpose(np.asarray(pred_list)), np.asarray(alpha))
        thresh = np.amin(temp_strong[0:n_pos])
        temp_strong_predict = np.zeros(temp_strong.shape)
        temp_strong_predict[temp_strong > thresh] = 1

        TPR = np.sum(temp_strong_predict[:n_pos]) / n_pos
        # FNR = 1 - TPR
        FPR = np.sum(temp_strong_predict[n_pos:]) / n_neg
        TNR = 1 - FPR

        if TPR >= tolerant_tpr and FPR <= tolerant_fpr:
            print("Tolerant classifier found! At iteration: %d\n" % itr)
            print("TPR = %f, FPR = %f"%(TPR, FPR))
            break

    neg_result = temp_strong_predict[n_pos:]
    neg_result_sort = np.sort(neg_result)
    neg_idx = np.argsort(neg_result)
    for j in range(n_neg):
        if neg_result_sort[j] > 0:
            neg_idx = neg_idx[j:n_neg]
            break

    total_iteration = itr + 1
    final_params = [ht, alpha, thresh, total_iteration]
    return final_params, TPR, TNR


def test_adaboost(feature_list, cascade_param_list):
    ht = cascade_param_list[0]
    alpha = cascade_param_list[1][1]
    n_weak_clf = ht[:][0]
    pred_list = list()
    for i in range(len(n_weak_clf)):
        x = n_weak_clf[i][0]
        polar = n_weak_clf[i][1]
        thresh = n_weak_clf[i][2]

        temp_feature = feature_list[x]
        if polar == 1:
            pred = np.array(temp_feature >= thresh)
        else:
            pred = np.array(temp_feature < thresh)
        pred_list.append(pred)

    pred_list = np.array(pred_list).astype(int)
    alpha = np.array(alpha)
    temp_strong_pred = np.matmul(np.transpose(pred_list), alpha)
    alpha_thresh = np.sum(alpha) / 2
    strong_predict = np.where(temp_strong_pred >= alpha_thresh, 1, 0)
    return strong_predict


if __name__ == '__main__':
    train_path = 'ECE661_2020_hw11_DB2[7700]\\train'
    test_path = 'ECE661_2020_hw11_DB2[7700]\\test'

    positive_train = load_img(os.path.join(train_path, 'positive'))
    negative_train = load_img(os.path.join(train_path, 'negative'))

    # transform img in positive fold to integral img
    positive_target = np.ones((1, len(positive_train)))
    positive_train_int = list()
    positive_feature_list = list()
    for img in positive_train:
        int_img = compute_integral(img)
        positive_train_int.append(int_img)
        positive_feature_list.append(generate_harr_feature(int_img))
    positive_feature_list = np.array(positive_feature_list)

    # transform img in negative fold to integral img
    negative_target = np.zeros((1, len(negative_train)))
    negative_train_int = list()
    negative_feature_list = list()
    for img in negative_train:
        int_img = compute_integral(img)
        negative_train_int.append(int_img)
        negative_feature_list.append(generate_harr_feature(int_img))
    negative_feature_list = np.array(negative_feature_list)

    total_feature = np.transpose(np.vstack((positive_feature_list, negative_feature_list)))
    # save as file
    # with open('total_train_feature.pickle', 'wb') as handle:
    #     pickle.dump(total_feature, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('positive_train_target.pickle', 'wb') as handle:
    #     pickle.dump(positive_target, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('negative_train_target.pickle', 'wb') as handle:
    #     pickle.dump(negative_target, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load data
    with open('total_train_feature.pickle', 'rb') as handle:
        total_feature = pickle.load(handle)
    with open('positive_train_target.pickle', 'rb') as handle:
        positive_target = pickle.load(handle)
    with open('negative_train_target.pickle', 'rb') as handle:
        negative_target = pickle.load(handle)

    tolerant_tpr = 1
    tolerant_fpr = 0.6
    stages = 15
    cascade_result = list()
    tpr_list = list()
    fpr_list = list()
    for st in range(stages):
        print("Stage %d" % st)
        stage_params, stage_tpr, stage_fpr = cascade_classifiers(total_feature, positive_target, negative_target,
                                                    tolerant_tpr, tolerant_fpr, n_iteration=100)
        # print("TPR=%f, FPR=%f" %(stage_tpr, stage_fpr))
        cascade_result.append(stage_params)
        tpr_list.append(stage_tpr)
        fpr_list.append(stage_fpr)

    with open('cascade_result.pickle', 'wb') as handle:
        pickle.dump(cascade_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.title('Training')
    x = range(stages)
    plt.xticks(x)
    plt.plot(x, tpr_list, '-o', label='tpr')
    plt.plot(x, fpr_list, '-^', label='fpr')
    plt.xlabel("Rate")
    plt.ylabel("stage")
    plt.show()

    #begin testing

    positive_test = load_img(os.path.join(test_path, 'positive'))
    negative_test = load_img(os.path.join(test_path, 'negative'))
    
    # transform img in positive fold to integral img
    positive_target_test = np.ones((1, len(positive_test)))
    positive_train_int = list()
    positive_feature_list = list()
    for img in positive_test:
        int_img = compute_integral(img)
        positive_train_int.append(int_img)
        positive_feature_list.append(generate_harr_feature(int_img))
    positive_feature_list = np.array(positive_feature_list)

    # transform img in negative fold to integral img
    negative_target_test = np.zeros((1, len(negative_test)))
    negative_train_int = list()
    negative_feature_list = list()
    for img in negative_test:
        int_img = compute_integral(img)
        negative_train_int.append(int_img)
        negative_feature_list.append(generate_harr_feature(int_img))
    negative_feature_list = np.array(negative_feature_list)

    total_feature = np.transpose(np.vstack((positive_feature_list, negative_feature_list)))
    # save as file
    with open('total_test_feature.pickle', 'wb') as handle:
        pickle.dump(total_feature, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('positive_test_target.pickle', 'wb') as handle:
        pickle.dump(positive_target_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('negative_test_target.pickle', 'wb') as handle:
        pickle.dump(negative_target_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load data
    with open('total_test_feature.pickle', 'rb') as handle:
        total_test_feature = pickle.load(handle)
    with open('positive_train_target.pickle', 'rb') as handle:
        positive_target_test = pickle.load(handle)
    with open('negative_train_target.pickle', 'rb') as handle:
        negative_target_test = pickle.load(handle)
    with open('cascade_result.pickle', 'rb') as handle:
        cascade_result = pickle.load(handle)

    y_test = np.hstack((positive_target_test, negative_target_test))
    # y_test = np.zeros((618,))
    # y_test[:179,]=1
    predict_test = test_adaboost(total_test_feature, cascade_result)
    fpr, tpr, threshold = metrics.roc_curve(y_test, predict_test)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Testing')
    plt.plot(fpr,  'b', label='fpr')
    plt.plot(tpr, label='tpr')
    plt.legend(loc='lower right')
    plt.ylabel('Rate')
    plt.show()

    pdb.set_trace()
