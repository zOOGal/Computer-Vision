import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt


def img_process(imgs_path):
    """

    :param imgs_path:
    :return: img_vec_list.shape=(16384, 630), true_labels.shape=(630,), cov.shape=(630, 630)
    """
    imgs_list = os.listdir(imgs_path)
    img_vec_list = list()
    true_labels = list()
    for img_name in imgs_list:
        img = cv2.imread(os.path.join(imgs_path, img_name), 0)  # read as gray-scale
        img_vector = img.ravel()
        img_vec_list.append(img_vector)
        true_labels.append(int(img_name[:2]))
    img_vec_list = np.array(img_vec_list)
    # composite matrix
    cov = np.cov(img_vec_list)
    # print("Covariance of %s is:\n"%imgs_path, cov)
    img_vec_list = np.transpose(img_vec_list)
    img_vec_list = img_vec_list / np.linalg.norm(img_vec_list, axis=0)
    mean = np.mean(img_vec_list, axis=1)
    img_vec_list = img_vec_list - mean[:, None]  # normalize the vector to make the sum=0
    print("Mean of %s is: \n" % imgs_path, mean)
    true_labels = np.array(true_labels)
    true_labels = np.transpose(true_labels)
    return img_vec_list, true_labels, cov, mean


def PCA(imgs_vec_matrix, imgs_cov):
    d, u = np.linalg.eig(imgs_cov)
    index_list = np.argsort(-1 * d)  # index with descending-value order
    u = u[:, index_list]
    w = np.matmul(imgs_vec_matrix, u)
    w = w / np.linalg.norm(w, axis=0)
    return w


def p_dimension_classify(X_train_raw, y_train, X_test_raw, y_test, w, p):
    """

    :param X_train_raw: all training features matrix
    :param y_train: training target
    :param X_test_raw:
    :param y_test:
    :param w: whole pca space
    :param p: the # of desired dimension
    :return:
    """
    subspace = w[:, :p + 1]
    X_train = np.matmul(X_train_raw, subspace)
    X_test = np.matmul(X_test_raw, subspace)

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train, y_train)

    y_predict = neigh.predict(X_test)
    acc = metrics.accuracy_score(y_pred=y_predict, y_true=y_test)
    print("Accuracy when p=%d: " % p, acc)
    return acc


def LDA(data, mean, n_class):
    class_len = int(data.shape[1] / n_class)

    class_mean = np.zeros((data.shape[0], n_class))
    class_diff = np.zeros(data.shape)

    for i in range(n_class):
        class_mean[:, i] = np.mean(data[:, i * class_len:(i + 1) * class_len], axis=1)
        class_diff[:, i * class_len:(i + 1) * class_len] = data[:, i * class_len:(i + 1) * class_len] - class_mean[:, i,
                                                                                                        None]

    class_mean -= mean[:, None]

    d, u = np.linalg.eig(np.matmul(np.transpose(class_mean), class_mean))
    index_list = np.argsort(-1 * d)  # index with descending-value order
    d = d[index_list]
    u = u[:, index_list]
    V = np.matmul(class_mean, u)  # eigenvectors of SB
    Db = np.eye(n_class) * (np.sqrt(d))
    Z = np.matmul(V, Db)
    X = np.matmul(np.transpose(Z), class_diff)
    dw, uw = np.linalg.eig(np.matmul(X, np.transpose(X)))
    index_list = np.argsort(dw)
    uw = uw[:, index_list]
    W = np.matmul(Z, uw)
    return W


if __name__ == '__main__':
    n_dimension = 20    # set the max p to try

    training_path = 'ECE661_2020_hw11_DB1\\train'
    testing_path = 'ECE661_2020_hw11_DB1\\test'

    training_list, training_target, training_cov, training_mean = img_process(training_path)
    testing_list, testing_target, testing_cov, testing_mean = img_process(testing_path)

    pca_w = PCA(training_list, training_cov)

    n_class = max(training_target)

    lda_w = LDA(training_list, training_mean, n_class)

    training_X = np.transpose(training_list)  # each column is a feature
    testing_X = np.transpose(testing_list)

    acc_pca_list = list()
    for i in range(n_dimension):
        acc = p_dimension_classify(training_X, training_target, testing_X, testing_target, pca_w, i)
        acc_pca_list.append(acc)

    acc_lda_list = list()
    for i in range(n_dimension):
        acc = p_dimension_classify(training_X, training_target, testing_X, testing_target, lda_w, i)
        acc_lda_list.append(acc)

    x = range(1, n_dimension + 1)
    plt.title('PCA/LDA as function of p&acc')
    plt.xticks(x)
    plt.plot(x, acc_pca_list, '-o', label='PCA')
    plt.plot(x, acc_lda_list, '-^', label='LDA')
    plt.xlabel("p")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    # for i, j in zip(x, acc_pca_list):
    #     plt.text(i, j, str(j)[:5], color="green", fontsize=10)
    # for i, j in zip(x, acc_lda_list):
    #     plt.text(i, j, str(j)[:5], color="green", fontsize=10)
    plt.show()
