import cv2
import numpy as np
import scipy.optimize
import pdb
import matplotlib.pyplot as plt


def normalize_coor(coor):
    x = coor[:, 0]
    y = coor[:, 1]
    dist_x = (x - np.mean(x)) ** 2
    dist_y = (y - np.mean(y)) ** 2
    dist = np.sqrt(dist_x + dist_y)
    dist_mean = np.mean(dist)
    scale = np.sqrt(2) / dist_mean
    a = -scale * np.mean(x)
    b = -scale * np.mean(y)
    trans = np.zeros((3, 3))
    trans[0][0] = scale
    trans[0][2] = a
    trans[1][1] = scale
    trans[1][2] = b
    trans[2][2] = 1
    return trans


# find fundamental matrix F
def estimate_F(coor1, coor2):
    pair_len = coor1.shape[0]
    A = np.zeros((pair_len, 9))

    for i in range(pair_len):
        A[i, 0] = coor2[i][0] * coor1[i][0]
        A[i, 1] = coor2[i][0] * coor1[i][1]
        A[i, 2] = coor2[i][0]
        A[i, 3] = coor2[i][1] * coor1[i][0]
        A[i, 4] = coor2[i][1] * coor1[i][1]
        A[i, 5] = coor2[i][1]
        A[i, 6] = coor1[i][0]
        A[i, 7] = coor1[i][1]
        A[i, 8] = 1

    u, s, v = np.linalg.svd(A)

    f = np.transpose(v[-1])
    F = np.reshape(f, (3, 3))
    u, s, v = np.linalg.svd(F)
    s[2] = 0
    F = np.matmul(u, np.matmul(np.diag(s), v))

    print('Raw F: ', F)
    return F


# this is to find p1 p2 e1 e2, F is the fundamental matrix
def find_p(F):
    u, s, v = np.linalg.svd(F)
    e1 = np.transpose(v)[:, -1]
    e2 = u[:, -1]
    p1 = np.zeros((3, 4))
    np.fill_diagonal(p1, 1)
    p2 = np.zeros((3, 4))
    F = F / F[2][2]
    p2[:, 0:3] = np.cross(e2, F)
    p2[:, 3] = e2
    # e1 = e1 / e1[2]
    # e2 = e2/ e2[2]
    # print('p1: ', p1)
    # print('p2: ', p2)
    return e1, e2, p1, p2


# this is for calculating world coordinated to reconstruct 3-D image
def find_world_coor(F, coor1, coor2):
    """

    :param F: F after refinement
    :param coor1:left coor
    :param coor2:right coor
    :return:
    """
    F = F.reshape((3, 3))
    p1, p2 = find_p(F)[2:4]
    pair_len = coor1.shape[0]
    world_coor = np.zeros((4, pair_len))

    for i in range(pair_len):
        A = np.zeros((4, 4))
        A[0] = coor1[i][0] * np.transpose(p1[2, :]) - p1[0, :]
        A[1] = coor1[i][1] * np.transpose(p1[2, :]) - p1[1, :]
        A[2] = coor2[i][0] * np.transpose(p2[2, :]) - p2[0, :]
        A[3] = coor2[i][1] * np.transpose(p2[2, :]) - p2[1, :]
        u, s, v = np.linalg.svd(A)
        # solution is the smallest eigenvector of A_T*A
        # world_coor[:, i] = v[-1, :] / v[-1, -1]
        world_coor[:, i] = v[-1, :]
        world_coor[:, i] /= np.linalg.norm(world_coor[:, i])
    return world_coor


# loss function for LM
def loss_Func(F, coor1, coor2):
    F = F.reshape((3, 3))
    p1, p2 = find_p(F)[2:4]
    pair_len = coor1.shape[0]
    world_coor = np.zeros((4, pair_len))

    for i in range(pair_len):
        A = np.zeros((4, 4))
        A[0] = coor1[i][0] * np.transpose(p1[2, :]) - p1[0, :]
        A[1] = coor1[i][1] * np.transpose(p1[2, :]) - p1[1, :]
        A[2] = coor2[i][0] * np.transpose(p2[2, :]) - p2[0, :]
        A[3] = coor2[i][1] * np.transpose(p2[2, :]) - p2[1, :]
        u, s, v = np.linalg.svd(A)
        # solution is the smallest eigenvector of A_T*A
        # world_coor[:, i] = v[-1, :] / v[-1, -1]
        world_coor[:, i] = v[-1, :]
        world_coor[:, i] /= np.linalg.norm(world_coor[:, i])

    # find predicted coor1&coor2
    pred_coor1 = np.matmul(p1, world_coor)
    pred_coor1 = pred_coor1 / pred_coor1[2]
    pred_coor2 = np.matmul(p2, world_coor)
    pred_coor2 = pred_coor2 / pred_coor2[2]

    pred_coor1 = np.transpose(pred_coor1)
    pred_coor2 = np.transpose(pred_coor2)
    # err_dist1 = pred_coor1 - np.transpose(coor1)
    # err_dist2 = pred_coor2 - np.transpose(coor2)

    # err_dist1 = (np.linalg.norm(pred_coor1 - np.transpose(coor1), axis=0))**2
    # err_dist2 = (np.linalg.norm(pred_coor2 - np.transpose(coor2), axis=0))**2
    err_dist1 = coor1 - pred_coor1
    err_dist2 = coor2 - pred_coor2
    for i in range(8):
        err_dist1[i] = np.linalg.norm(err_dist1[i]) ** 2
        err_dist2[i] = np.linalg.norm(err_dist2[i]) ** 2
    error = np.concatenate((err_dist2.flatten(), err_dist1.flatten()))
    return error.flatten()


def nonlinear_LM(F, coor1, coor2):
    f_init = [F[0][0], F[0][1], F[0][2], F[1][0],
              F[1][1], F[1][2], F[2][0], F[2][1], F[2][2]]

    sol = scipy.optimize.least_squares(loss_Func, f_init, method='lm', args=[coor1, coor2])
    F_refined = np.array([[sol.x[0], sol.x[1], sol.x[2]],
                          [sol.x[3], sol.x[4], sol.x[5]],
                          [sol.x[6], sol.x[7], sol.x[8]]])
    print('Refined F: ', F_refined)
    return F_refined


def rectify_img(F, e1, e2, p1, p2, img1, img2, coor1, coor2):
    height = img1.shape[0]
    width = img1.shape[1]

    theta = np.arctan((e2[1] - height / 2) / (width / 2 - e2[0]))
    focal = np.cos(theta) * (e2[0] - width / 2) - np.sin(theta) * (e2[1] - height / 2)

    G = np.zeros((3, 3))  # homography taking epipole to [1,0,0]
    np.fill_diagonal(G, 1)
    G[2][0] = -1 / focal

    R = np.zeros((3, 3))  # rotation matrix
    R[0][0] = np.cos(theta)
    R[0][1] = -np.sin(theta)
    R[1][0] = np.sin(theta)
    R[1][1] = np.cos(theta)
    R[2][2] = 1

    T = np.zeros((3, 3))  # homography to translate the 2nd image center to origin
    np.fill_diagonal(T, 1)
    T[0][2] = -width / 2
    T[1][2] = -height / 2

    mul1 = np.matmul(G, R)
    H2 = np.matmul(mul1, T)
    image_center = np.array([width / 2, height / 2, 1])
    rectified_center = np.matmul(H2, np.transpose(image_center))
    rectified_center = rectified_center / rectified_center[2]

    T2 = np.zeros((3, 3))
    np.fill_diagonal(T2, 1)
    T2[0][2] = width / 2 - rectified_center[0]
    T2[1][2] = height / 2 - rectified_center[1]

    H2 = np.matmul(T2, H2)  # overall h for img2

    E = np.transpose(np.vstack([e2, e2, e2]))
    F = F / F[2][2]
    M = np.cross(e2, F) + E

    H0 = np.matmul(H2, M)

    new_coor1 = np.zeros((8, 3))
    new_coor2 = np.zeros((8, 3))

    for i in range(8):
        new_coor1[i] = np.transpose(np.matmul(H0, np.transpose(coor1[i])))
        new_coor1[i] = new_coor1[i] / new_coor1[i][2]
        new_coor2[i] = np.transpose(np.matmul(H2, np.transpose(coor2[i])))
        new_coor2[i] = new_coor2[i] / new_coor2[i][2]

    A = new_coor1
    b = new_coor2[:, 0]
    x = np.matmul(np.linalg.pinv(A), b)

    HA = np.zeros((3, 3))
    np.fill_diagonal(HA, 1)
    HA[0][0] = x[0]
    HA[0][1] = x[1]
    HA[0][2] = x[2]
    H1 = np.matmul(HA, H0)

    rectified_center = np.matmul(H1, image_center)
    rectified_center = rectified_center / rectified_center[2]
    T1 = np.zeros((3, 3))
    np.fill_diagonal(T1, 1)
    T1[0][2] = width / 2 - rectified_center[0]
    T1[1][2] = height / 2 - rectified_center[1]

    H1 = np.matmul(T1, H1)
    F_rectified = np.matmul(np.matmul(np.linalg.pinv(np.transpose(H2)), F), np.linalg.inv(H1))

    coor_img1 = np.zeros((8, 3))
    coor_img2 = np.zeros((8, 3))
    for i in range(8):
        coor_img1[i] = np.matmul(H1, coor1[i])
        coor_img1[i] = coor_img1[i] / coor_img1[i][2]
        coor_img2[i] = np.matmul(H2, coor2[i])
        coor_img2[i] = coor_img2[i] / coor_img2[i][2]

    u, s, v = np.linalg.svd(F_rectified)
    e1_rectified = v[-1:0]
    e2_rectified = u[:, -1]

    H1 = H1 / H1[2][2]
    H2 = H2 / H2[2][2]
    return coor_img1, coor_img2, H1, H2, F_rectified, e1_rectified, e2_rectified


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

    height_min = min(new_roi_set[1])
    height_max = max(new_roi_set[1])
    width_min = min(new_roi_set[0])
    width_max = max(new_roi_set[0])

    # scaling
    s = np.zeros((3, 3))
    s[2, 2] = 1
    s[0, 0] = img.shape[1] / (width_max - width_min) / 2
    s[1, 1] = img.shape[0] / (height_max - height_min) / 2
    h = np.matmul(s, h)

    new_roi_set = np.matmul(h, roi_set)
    for i in range(4):
        new_roi_set[:, i] = new_roi_set[:, i] / new_roi_set[2][i]

    new_roi_set = new_roi_set.astype(int)

    return new_roi_set, h


# modified function from hw5(combine image)
def single_img(final_roi):
    height_min = min(final_roi[1])
    height_max = max(final_roi[1])
    width_min = min(final_roi[0])
    width_max = max(final_roi[0])

    final_height = int(height_max - height_min + 1)
    final_width = int(width_max - width_min + 1)
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


def canny_edges(img, img_num):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 200, 700, apertureSize=3)
    cv2.imwrite('img%d_canny.jpg' % img_num, edges)

    return edges


def find_coor(img1, img2, window_size=3, mode='SSD'):
    edge1 = canny_edges(img1, 1)
    edge2 = canny_edges(img2, 2)

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if mode == 'SSD':
        # matched_pair is to save the points got paired from corresponding points in img2
        match_pair = list()
        for w in range(5, edge1.shape[0] - 5):
            for h in range(5, edge1.shape[1] - 5):
                ssd = list()
                if edge1[w][h] == 255:
                    threshold = 2000
                    for h2 in range(5, edge2.shape[1] - 5):
                        dist = gray_img1[w - window_size:w + window_size + 1, h - window_size:h + window_size + 1] \
                               - gray_img2[w - window_size:w + window_size + 1, h2 - window_size:h2 + window_size + 1]
                        dist_sum = np.sum(dist ** 2)
                        ssd.append(dist_sum)
                    if min(ssd) < threshold:
                        ssd = np.array(ssd)
                        temp = np.where(ssd == min(ssd))[0][0]
                        match_pair.append([np.array([h, w]), np.array([temp, w])])

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


if __name__ == '__main__':
    img1 = cv2.imread('izze1.jpg')
    img2 = cv2.imread('izze2.jpg')

    coor1 = [[549, 196, 1],
             [185, 369, 1],
             [229, 680, 1],
             [550, 1065, 1],
             [990, 513, 1],
             [630, 768, 1],
             [425, 513, 1],
             [567, 779, 1]]
    coor1 = np.array(coor1).astype(float)
    trans1 = normalize_coor(coor1)
    norm_coor1 = np.transpose(np.matmul(trans1, np.transpose(coor1)))

    coor2 = [[469, 244, 1],
             [67, 425, 1],
             [150, 738, 1],
             [448, 1145, 1],
             [898, 579, 1],
             [466, 855, 1],
             [282, 574, 1],
             [415, 862, 1]]
    coor2 = np.array(coor2).astype(float)
    trans2 = normalize_coor(coor2)
    norm_coor2 = np.transpose(np.matmul(trans2, np.transpose(coor2)))

    raw_f = estimate_F(norm_coor1, norm_coor2)
    f = np.matmul(np.transpose(trans2), np.matmul(raw_f, trans1))

    # ====================================using f without LM========================================
    e1, e2, p1, p2 = find_p(f)
    coor_img1, coor_img2, H1, H2, F_rectified, e1_rectified, e2_rectified = rectify_img(f, e1, e2, p1, p2,
                                                                                        img1, img2, coor1, coor2)
    new_roi1, h1 = find_new_roi_set(img1, H1)
    new_roi2, h2 = find_new_roi_set(img2, H2)

    final_img1, height_min1, width_min1 = single_img(new_roi1)
    out1 = mapping(h1, img1, final_img1, height_min1, width_min1)
    cv2.imwrite('out1.jpg', out1)

    final_img2, height_min2, width_min2 = single_img(new_roi2)
    out2 = mapping(h2, img2, final_img2, height_min2, width_min2)
    cv2.imwrite('out2.jpg', out2)

    # ================================= using f with Lm==============================================
    F_refined = nonlinear_LM(f, coor1, coor2)
    e3, e4, p3, p4 = find_p(F_refined)
    coor_img1, coor_img2, H3, H4, F_refined_rectified, e1_rectified, e2_rectified = rectify_img(F_refined, e3, e4, p3, p4,
                                                                                        img1, img2, coor1, coor2)
    new_roi3, h3 = find_new_roi_set(img1, H3)
    new_roi4, h4 = find_new_roi_set(img2, H4)

    final_img1, height_min1, width_min1 = single_img(new_roi3)
    out3 = mapping(h3, img1, final_img1, height_min1, width_min1)
    cv2.imwrite('out3.jpg', out3)

    final_img2, height_min2, width_min2 = single_img(new_roi4)
    out4 = mapping(h4, img2, final_img2, height_min2, width_min2)
    cv2.imwrite('out4.jpg', out4)

    rec1 = cv2.imread('rectified1.jpg')
    rec2 = cv2.imread('rectified2.jpg')

    match_pair = find_coor(rec1, rec2)
    paired_img = mark_pair(rec1, rec2, match_pair)
    cv2.imwrite('paired_output.jpg', paired_img)

    lines = [[1, 2], [1, 3], [3, 4], [2, 4], [1, 5], [3, 6], [5, 6], [6, 7], [4, 7]]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca(projection='3d')
    Labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

    world = find_world_coor(F_refined, coor1, coor2)
    for idx in range(8):
        plt.scatter(world[0, idx], world[1, idx], world[2, idx])
        ax.text(world[0, idx], -world[1, idx], -world[2, idx], Labels[idx], zdir=None)
    for line in lines:
        Xs = [world[0, idx - 1] for idx in line]
        Ys = [-world[1, idx - 1] for idx in line]  # Rotate y so that it matches original direction
        Zs = [-world[2, idx - 1] for idx in line]  # Rotate z so that it matches original direction
        plt.plot(Xs, Ys, Zs)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.savefig("3d_left.png")
    ax.view_init(azim=-70)
    pdb.set_trace()
