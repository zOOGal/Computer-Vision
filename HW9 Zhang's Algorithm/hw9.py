import cv2
import numpy as np
import os
import glob
import pdb
import scipy.optimize


def Hough_Canny_line(img, img_name):
    '''
    ref:https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    :param img:
    :return:
    '''
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 3500, 4000, apertureSize=5)
    cv2.imwrite(img_name + 'canny.jpg', edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 40)  # hough_thresh=50

    draw_lines(lines, img, 'raw', img_name)

    return lines


def draw_lines(lines, img, mode, img_name):
    line_image = np.copy(img)  # creating an image copy to draw lines on
    if lines is None:
        print("No line detected!\n")
    else:
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)

                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(img_name + 'addLines%s.jpg' % mode, line_image)
    return


def find_intersection(line1, line2, mode='2D'):
    inter = np.cross(line1, line2)
    if mode == '2D':
        return inter
    elif mode == '3D':
        inter[0] = inter[0] / inter[2]
        inter[1] = inter[1] / inter[2]
        return np.array([inter[0], inter[1]])


def del_redundant_lines(lines, dist_thresh=20):
    final_lines = list()
    for i in range(len(lines)):
        line = lines[i]
        if i == 0:
            final_lines.append(line)
        else:
            temp_rho = line[0]
            '''
            Since the [lines] is already sorted, compare the temp_element with the last one in
            final_lines is enough 
            '''
            if abs(temp_rho - final_lines[-1][0]) > dist_thresh:
                final_lines.append(line)
            else:
                pass

    return final_lines


def split_lines(lines, img, img_name):
    '''
    split lines into vertical lines & horizontal lines
    :param img:
    :param lines:
    :return:
    '''
    thetas = lines[:, :, 1]
    horizon_index = np.where(abs(np.tan(thetas)) > 1)
    vertical_index = np.where(abs(np.tan(thetas)) <= 1)
    horizon = lines[horizon_index]
    vertical = lines[vertical_index]
    horizon = sorted(horizon, key=lambda x: x[0] * np.sin(x[1]))
    final_horizon = del_redundant_lines(horizon)
    vertical = sorted(vertical, key=lambda x: x[0] * np.cos(x[1]))
    final_vertical = del_redundant_lines(vertical)
    final_lines = final_horizon + final_vertical
    # print('Total number of lines is:', len(final_lines))
    final_lines = np.array(final_lines)
    # reshape lines into (n,1,2) to draw
    final_lines = final_lines.reshape((final_lines.shape[0], 1, final_lines.shape[1]))
    draw_lines(final_lines, img, 'clean', img_name)

    # final_horizon = np.array(final_horizon)
    # final_horizon = final_horizon.reshape((final_horizon.shape[0], 1, final_horizon[1])
    return final_horizon, final_vertical


def convert2HC(line):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)

    hc1 = np.array([x1, y1, 1])
    hc2 = np.array([x2, y2, 1])

    hc = np.cross(hc1, hc2)
    hc = hc / hc[2]

    return hc


def find_corners(horizon, vertical, img, img_name):
    corners = list()
    count = 0
    for hl in horizon:
        hl = convert2HC(hl)
        for vl in vertical:
            vl = convert2HC(vl)
            coor = find_intersection(hl, vl, '3D')
            corners.append(coor)
            draw_corners(coor, count, img)
            count += 1

    cv2.imwrite(img_name + 'marked.jpg', img)
    # print(corners)

    return corners


def draw_corners(coor, count, img, color=(255, 0, 0)):
    '''

    :param coor: coordinated of one corner
    :param count: corner's label
    :return:
    '''
    x = int(coor[0])
    y = int(coor[1])
    new_coor = (x, y)
    cv2.circle(img, new_coor, 1, color, 2)
    cv2.putText(img, str(count), new_coor, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
    return


def Homography(world_coor, image_coor):
    n = len(image_coor)
    A = np.zeros((2 * n, 9))

    for i in range(n):
        A[2 * i, :] = [world_coor[i][0], world_coor[i][1], 1, 0, 0, 0,
                       -world_coor[i][0] * image_coor[i][0],
                       -world_coor[i][1] * image_coor[i][0], -image_coor[i][0]]
        A[2 * i + 1, :] = [0, 0, 0, world_coor[i][0], world_coor[i][1], 1,
                           -world_coor[i][0] * image_coor[i][1],
                           -world_coor[i][1] * image_coor[i][1], -image_coor[i][1]]

    # svd composition
    u, s, v = np.linalg.svd(A)
    upper_h = v[8]

    h = upper_h.reshape((3, 3))
    h = (h / h[2, 2])
    return h


def cal_v(H, i, j):
    v_ij = np.array([H[0, i] * H[0, j],
                     H[0, i] * H[1, j] + H[1, i] * H[0, j],
                     H[1, i] * H[1, j],
                     H[2, i] * H[0, j] + H[0, i] * H[2, j],
                     H[2, i] * H[1, j] + H[1, i] * H[2, j],
                     H[2, i] * H[2, j]])

    return v_ij


def find_intrinsic(corner_group, world_coord):
    h_group = list()
    V = list()
    for corners in corner_group:
        h = Homography(world_coord, corners)
        h_group.append(h)
        v_11 = cal_v(h, 0, 0)
        v_22 = cal_v(h, 1, 1)
        v_12 = cal_v(h, 0, 1)
        V.append(v_12)
        V.append(v_11 - v_22)
    V = np.asarray(V)
    u, s, v = np.linalg.svd(V)
    upper_w = v[-1]
    w11 = upper_w[0]
    w12 = upper_w[1]
    w22 = upper_w[2]
    w13 = upper_w[3]
    w23 = upper_w[4]
    w33 = upper_w[5]
    # print(upper_w)
    x0 = (w12 * w13 - w11 * w23) / (w11 * w22 - w12 ** 2)
    lambda_ = w33 - (w13 ** 2 + x0 * (w12 * w13 - w11 * w23)) / w11
    # print(lambda_)
    alpha_x = np.sqrt(lambda_ / w11)
    alpha_y = np.sqrt((lambda_ * w11) / (w11 * w22 - w12 ** 2))
    s = -(w12 * (alpha_x ** 2) * alpha_y) / lambda_
    y0 = (s * x0 / alpha_y) - (w13 * (alpha_x ** 2) / lambda_)
    k = np.zeros((3, 3))
    k[0, 0] = alpha_x
    k[0, 1] = s
    k[0, 2] = x0
    k[1, 1] = alpha_y
    k[1, 2] = y0
    k[2, 2] = 1
    print('intrinsic parameter:\n', k)
    return k, h_group


def find_extrinsic(h, k):
    '''

    :param h: estimated homography
    :param k: intrinsic parameter
    :return:
    '''
    h1 = h[:, 0]
    h2 = h[:, 1]
    h3 = h[:, 2]
    k_inv = np.linalg.inv(k)
    r1 = np.matmul(k_inv, h1)
    factor = 1 / np.linalg.norm(r1)
    r1 = factor * r1
    r2 = np.matmul(k_inv, h2) * factor
    r3 = np.cross(r1, r2)
    t = np.matmul(k_inv, h3) * factor
    ex = np.transpose(np.vstack((r1, r2, t)))
    print('Extrinsic matrix:\n', ex)
    return ex


def loss_Func(h, world_coor, image_coor):
    """

    :param h: h=intrinsic*extrinsic
    :param image_coor:
    :param world_coor:
    :return:
    """
    new_world_coor = np.ones((3, len(world_coor)))
    new_world_coor[0][:] = [x[0] for x in world_coor]
    new_world_coor[1][:] = [x[1] for x in world_coor]
    h_trans = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]])

    predict_image_coor = np.matmul(h_trans, new_world_coor)
    predict_image_coor = predict_image_coor[0:3, :] / predict_image_coor[2, :]
    # print(predict_image_coor[2])
    predict_image_coor = predict_image_coor[:2]

    error = image_coor - np.transpose(predict_image_coor)

    return error.flatten()


def nonlinear_LM(camera_h, image_coor, world_coor):
    h_init = [camera_h[0][0], camera_h[0][1], camera_h[0][2], camera_h[1][0],
              camera_h[1][1], camera_h[1][2], camera_h[2][0], camera_h[2][1], h[2][2]]

    sol = scipy.optimize.least_squares(loss_Func, h_init, method='lm', args=[world_coor, image_coor])
    h_refined = np.array([[sol.x[0], sol.x[1], sol.x[2]],
                          [sol.x[3], sol.x[4], sol.x[5]],
                          [sol.x[6], sol.x[7], sol.x[8]]])

    return h_refined


def draw_mapping(h, world_coor, image_coor, img):
    img_copy = img.copy()
    new_world_coor = np.ones((3, len(world_coor)))
    new_world_coor[0][:] = [x[0] for x in world_coor]
    new_world_coor[1][:] = [x[1] for x in world_coor]
    predict_image_coor = np.matmul(h, new_world_coor)
    predict_image_coor = predict_image_coor / predict_image_coor[2]
    predict_image_coor = np.transpose(predict_image_coor)[:, 0:2]
    error = predict_image_coor - image_coor
    squre_err = np.linalg.norm(error, axis=1)
    mean = np.mean(squre_err)
    var = np.var(squre_err)
    for i in range(predict_image_coor.shape[0]):
        coor = predict_image_coor[i]
        draw_corners(coor, i, img_copy, color=(0, 255, 0))

    return img_copy, mean, var


def reprojct(proj_coor, fixed_coor, camera_h):
    h = Homography(proj_coor, fixed_coor)  # homography from proj_coor to fixed coor
    reprojct_h = np.matmul(h, camera_h)
    return reprojct_h


if __name__ == '__main__':
    world_coordinates = []
    for x in range(10):
        for y in range(8):
            world_coordinates.append([x, y])

    data_path = 'Dataset2'
    imgs_path = glob.glob(os.path.join(data_path, '*'))
    # fixed_img_path = glob.glob(os.path.join(data_path, 'Pic_11.jpg'))

    corner_group = list()
    img_group = list()
    img_group_name = list()
    for img_name in imgs_path:
        img_color = cv2.imread(img_name)
        lines = Hough_Canny_line(img_color, img_name)
        horizon, vertical = split_lines(lines, img_color, img_name)
        if len(horizon) == 10 and len(vertical) == 8:
            corner = find_corners(horizon, vertical, img_color, img_name)
            corner_group.append(corner)
            img_group.append(img_color)
            img_group_name.append(img_name)
    print(img_group_name)
    print(len(corner_group))

    k, h_group = find_intrinsic(corner_group, world_coordinates)
    h_group = np.array(h_group)
    index = [3, 4, 6]  # choose images for calculating extrinsic and refine
    project_img = img_group[1]
    for idx in index:
        h = h_group[idx]
        ex = find_extrinsic(h, k)  # extrinsic matrix
        camera_h = np.matmul(k, ex)
        roproject_h = reprojct(corner_group[idx], corner_group[0], camera_h)
        # print('homo without LM: \n', camera_h)
        camera_refined_h = nonlinear_LM(camera_h, corner_group[idx], world_coordinates)
        reproject_refined_h = reprojct(corner_group[idx], corner_group[0], camera_refined_h)
        # print('homo after LM: \n', camera_refined_h)

        cv2.imwrite('before_project%d.jpg' % idx, project_img)
        camera_raw_map, raw_mean, raw_var = draw_mapping(roproject_h, world_coordinates, corner_group[idx], project_img)
        print('raw_mean: ', raw_mean)
        print('raw variance: ', raw_var)
        cv2.imwrite('raw_project%d.jpg' % idx, camera_raw_map)

        camera_refine_map, refined_mean, refined_var = draw_mapping(reproject_refined_h, world_coordinates,
                                                                    corner_group[idx], project_img)
        print('refined_mean: ', refined_mean)
        print('refined_var: ', refined_var)
        cv2.imwrite('refined_project%d.jpg' % idx, camera_refine_map)

    pdb.set_trace()
