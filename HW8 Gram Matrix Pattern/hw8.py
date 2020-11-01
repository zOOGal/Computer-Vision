import numpy as np
import cv2
import random
import glob
import os
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def generate_op(m=3):
    op = list()
    for i in range(m ** 2 - 1):
        op.append(random.uniform(-1, 1))
    op = op + [-sum(op)]
    op = np.array(op)
    op.reshape((m, m))
    return op


def convolution(img_path, op, shrink=16):
    try:
        img = cv2.imread(img_path, 0)  # as gray-scale
        img = cv2.resize(img, (shrink, shrink))
    except Exception as e:
        print(img_path)
    result = cv2.filter2D(img, -1, op)
    return result.reshape(-1)


def gram_matrix(img_feature_list, channel_num):
    '''

    :param img_feature_list: feature list for an image
    :param channel_num: = len(feature list)
    :return: vectorized gram-matrix's upper-triangular
    '''
    gram = np.zeros((channel_num, channel_num))
    for i in range(channel_num):
        for j in range(channel_num):
            gram[i][j] = np.sum(img_feature_list[i] * img_feature_list[j])
    gram_vector = gram[np.triu_indices(channel_num)]
    return gram_vector


def add_tag(data):
    tag_list = list()
    vec_list = list()
    for img, vec in data.items():
        if 'cloudy' in img:
            tag = 0
            tag_list.append(tag)
        if 'rain' in img:
            tag = 1
            tag_list.append(tag)
        if 'shine' in img:
            tag = 2
            tag_list.append(tag)
        if 'sunrise' in img:
            tag = 3
            tag_list.append(tag)
        vec_list.append(vec)

    tag_list = np.array(tag_list)
    vec_list = np.array(vec_list, dtype=np.float32)
    return tag_list, vec_list


def cal_accuracy(real_tag, predict_tag):
    match = sum(a == b for a, b in zip(real_tag, predict_tag))
    total = len(real_tag)
    return (match / total)[0]


if __name__ == '__main__':
    mode = 'train'
    channel_num = 3

    if mode == 'train':
        trial_num = 100
        best_accuracy = -1

        training_path = 'imagesDatabaseHW8\\training'
        imgs_path = glob.glob(os.path.join(training_path, '*'))
        training = {os.path.basename(img): None for img in imgs_path}

        for trial in range(trial_num):
            print('training')
            operator_list = list()
            for c in range(channel_num):
                op = generate_op()
                operator_list.append(op)
            for img in imgs_path:
                feature_list = list()
                for op in operator_list:
                    conv = convolution(img, op)
                    feature_list.append(conv)
                gram_vec = gram_matrix(feature_list, channel_num)
                training[os.path.basename(img)] = gram_vec
            tag, training_data = add_tag(training)

            X_train, X_validate, y_train, y_validate = \
                train_test_split(training_data, tag, test_size=0.3, random_state=50)

            svm = cv2.ml.SVM_create()
            svm.setType(cv2.ml.SVM_C_SVC)
            svm.setKernel(cv2.ml.SVM_LINEAR)
            svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
            svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
            print('predicting')
            predict_validate_tag = svm.predict(X_validate)[1]
            accuracy = cal_accuracy(y_validate, predict_validate_tag)
            print(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_operator = operator_list
                svm.save('best_model.xml')
                f = open("best_operator.obj", "wb")
                pickle.dump(best_operator, f)
                f.close()

        print('Best accuracy:', best_accuracy)
        print('best operator:', best_operator)

    if mode == 'test':
        testing_path = 'imagesDatabaseHW8\\testing'
        test_imgs_path = glob.glob(os.path.join(testing_path, '*'))
        testing = {os.path.basename(img): None for img in test_imgs_path}
        print('testing')

        if not os.path.isfile('best_model.xml'):
            print('Cannot locate the best model! Train first!\n')
            exit()
        else:
            svm = cv2.ml.SVM_load('best_model.xml')

        if not os.path.exists("best_operator.obj"):
            print("Find operator first!\n")
            exit()
        f1 = open("best_operator.obj", 'rb')
        operator_list = pickle.load(f1)
        f1.close()

        for img in test_imgs_path:
            feature_list = list()
            for op in operator_list:
                conv = convolution(img, op)
                feature_list.append(conv)
            gram_vec = gram_matrix(feature_list, channel_num)
            testing[os.path.basename(img)] = gram_vec

        test_true_tag, testing_data = add_tag(testing)
        predict_tag = svm.predict(testing_data)[1]
        accuracy = cal_accuracy(test_true_tag, predict_tag)
        print('Accuracy for testing data is:', accuracy)

        cf = confusion_matrix(test_true_tag, predict_tag)
        labels = ['cloudy', 'rain', 'shine', 'sunrise']
        sns.heatmap(cf, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.ylabel('Actual name')
        plt.xlabel('Prediction')
        # plt.show()
        plt.savefig('best_score.jpg')
        plt.close()
