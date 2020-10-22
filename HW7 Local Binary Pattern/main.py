import cv2
import pickle
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from BitVector import BitVector
from sklearn.metrics import confusion_matrix
import seaborn as sns


def bi_inter(A, B, C, D, du, dv):
    a = (1 - du) * (1 - dv) * A
    b = (1 - du) * dv * B
    c = du * (1 - dv) * C
    d = du * dv * D
    return a + b + c + d


def find_pattern(A, du=np.sqrt(2)/2, dv=np.sqrt(2)/2):
    '''

    :param A: 3 by 3 matrix, A(1,1) is the center
    :param du:
    :param dv:
    :return: pattern like fig.8 in Texture&Color page33
    '''
    pattern = list()
    up_l = bi_inter(A[1, 1], A[1, 0], A[0, 1], A[0, 0], du, dv)
    up_r = bi_inter(A[1, 1], A[1, 2], A[0, 1], A[0, 2], du, dv)
    down_l = bi_inter(A[1, 1], A[1, 0], A[2, 1], A[2, 0], du, dv)
    down_r = bi_inter(A[1, 1], A[1, 2], A[2, 1], A[2, 2], du, dv)
    neighbors = [A[2, 1], down_r, A[1, 2], up_r, A[0, 1], up_l, A[1, 0], down_l]
    for neighbor in neighbors:
        if neighbor >= A[1][1]:
            pattern.append(1)
        else:
            pattern.append(0)

    return pattern


def lbp(pattern, p=8):
    bit_vec = BitVector(bitlist=pattern)
    intvals = [int(bit_vec << 1) for _ in range(p)]
    min_bit_vec = BitVector(intVal=min(intvals), size=p)
    bit_vec_runs = min_bit_vec.runs()

    if len(bit_vec_runs) > 2:
        encoding = p + 1
    elif len(bit_vec_runs) == 1 and bit_vec_runs[0][0] == '1':
        encoding = p
    elif len(bit_vec_runs) == 1 and bit_vec_runs[0][0] == '0':
        encoding = 0
    else:
        encoding = len(bit_vec_runs[1])

    return encoding


def lbp_hist(img_path, r=1, p=8):
    img = cv2.imread(img_path, 0)  # read as grayscale
    hist = {t: 0 for t in range(p + 2)}
    max_width = img.shape[0] - r
    max_height = img.shape[1] - r
    for w in range(r, max_width):
        for h in range(r, max_height):
            frame = img[w - 1:w + 2, h - 1:h + 2]
            pattern = find_pattern(frame)
            encode_val = lbp(pattern, p)
            hist[encode_val] += 1

    return hist


def knn(test_data, train_class_data, dist_metric='Euclidean', k=5):
    '''

    :param test_data: 1-D
    :param train_class_data: 3-D: 5*20*10
    :param dist_metric:
    :param k:
    :return:
    '''
    if dist_metric == 'Euclidean':
        dist = np.linalg.norm(train_class_data - test_data, axis=2)
    if dist_metric == 'Manhattan':
        dist = np.sum(abs(train_class_data - test_data), axis=2)
    if dist_metric == 'Chebychev':
        dist = np.amax(abs(train_class_data - test_data), axis=2)

    new_dist = dist.ravel()
    index_list = new_dist.argsort()[:k]

    index_list = (index_list/20).astype(int)
    index_list = index_list.tolist()
    index = max(set(index_list), key=index_list.count)

    return index


def draw_confusion_matrix(testing_data, prediction_index):
    actual_index = list()
    for actual in testing_data.keys():
        if 'beach' in actual:
            actual_index.append(0)
        if 'building' in actual:
            actual_index.append(1)
        if 'car' in actual:
            actual_index.append(2)
        if 'mountain' in actual:
            actual_index.append(3)
        if 'tree' in actual:
            actual_index.append(4)
    cf = confusion_matrix(actual_index, prediction_index)
    labels = ['beach', 'building', 'car', 'mountain', 'tree']
    sns.heatmap(cf, annot=True, cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Actual name')
    plt.ylabel('Prediction')
    plt.show()


if __name__ == '__main__':
    runType = 'test'

    if runType == 'train':
        print('Training...\n')
        training_path = 'imagesDatabaseHW7\\training'
        class_list = os.listdir(training_path)
        class_name = {name: [] for name in class_list}

        for name in class_name:
            img_dir = os.path.join(training_path, name)
            imgs_path = glob.glob(os.path.join(img_dir, '*.jpg'))
            for img_path in imgs_path:
                hist = lbp_hist(img_path)
                temp_sum = sum(hist.values())
                percent_hist = {k: v / temp_sum for k, v in hist.items()}
                class_name[name].append(percent_hist)
                # class_name[name] = {k: class_name[name].get(k, 0) + percent_hist.get(k, 0) for k in set(percent_hist)}
            # temp_sum = sum(class_name[name].values())
            # class_name[name] = {k: v / temp_sum for k, v in class_name[name].items()}
            # plt.bar(class_name[name].keys(), class_name[name].values(), color='g')
            # plt.title(name + ' features')
            # plt.show()
        print('Histogram generated \n')

        f = open("Training.obj", "wb")
        pickle.dump(class_name, f)
        f.close()


    elif runType == 'test':
        print('Predicting...\n')
        testing_path = 'imagesDatabaseHW7\\testing'
        imgs_path = glob.glob(os.path.join(testing_path, '*.jpg'))
        predict = {os.path.basename(img): None for img in imgs_path}

        for img in imgs_path:
            predict_hist = lbp_hist(img)
            temp_sum = sum(predict_hist.values())
            predict_hist_percent = {k: v / temp_sum for k, v in predict_hist.items()}
            predict[os.path.basename(img)] = predict_hist_percent
        print('Prediction histogram generated \n')

        f2 = open("Testing.obj", "wb")
        pickle.dump(predict, f2)
        f2.close()

        if not os.path.exists("Training.obj"):
            print("Train first!\n")
            exit()
        f1 = open("Training.obj", 'rb')
        training = pickle.load(f1)
        f1.close()

        beach = list()
        building = list()
        car = list()
        mountain = list()
        tree = list()
        for i in range(len(training['beach'])):
            beach.append(list(training['beach'][i].values()))
            building.append(list(training['building'][i].values()))
            car.append(list(training['car'][i].values()))
            mountain.append(list(training['mountain'][i].values()))
            tree.append(list(training['tree'][i].values()))
        beach = np.array(beach)
        building = np.array(building)
        car = np.array(car)
        mountain = np.array(mountain)
        tree = np.array(tree)

        training_class = np.stack([beach, building, car, mountain, tree], axis=0)

        if not os.path.exists("Testing.obj"):
            print("No Testing files found!\n")
            exit()
        f3 = open("Testing.obj", 'rb')
        testing = pickle.load(f3)
        f3.close()

        prediction_index = list()
        for img, img_hist in testing.items():
            img_feature = np.array(list(testing[img].values()))
            img_index = knn(img_feature, training_class, dist_metric='Euclidean', k=5)
            if img_index == 0:
                testing[img] = 'beach'
                prediction_index.append(img_index)
            elif img_index == 1:
                testing[img] = 'building'
                prediction_index.append(img_index)
            elif img_index == 2:
                testing[img] = 'car'
                prediction_index.append(img_index)
            elif img_index == 3:
                testing[img] = 'mountain'
                prediction_index.append(img_index)
            elif img_index == 4:
                testing[img] = 'tree'
                prediction_index.append(img_index)
            print(img, testing[img], '\n')

        draw_confusion_matrix(testing, prediction_index)
    else:
        print('Cannot identify runType!\n')
