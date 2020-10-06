import numpy as np
import cv2 as cv2
import pdb


def cal_homography(p, q, r, s, frame_num, cor_object):
    a = np.zeros((8, 8))  # initiate the multiplied matrix a

    for i in range(4):
        if i == 0:
            plane_point = p
        elif i == 1:
            plane_point = q
        elif i == 2:
            plane_point = r
        else:
            plane_point = s

        a[2 * i, :] = [cor_object[i][0], cor_object[i][1], 1, 0, 0, 0,
                       -cor_object[i][0] * plane_point[frame_num-1][0],
                       -cor_object[i][1] * plane_point[frame_num-1][0]]  # odd row of a
        a[2 * i + 1, :] = [0, 0, 0, cor_object[i][0], cor_object[i][1], 1,
                           -cor_object[i][0] * plane_point[frame_num-1][1],
                           -cor_object[i][1] * plane_point[frame_num-1][1]]  # even row of a

    a_inverse = np.linalg.inv(a)  # the inverse of a

    b = np.zeros((8, 1))
    for i in range(4):
        if i == 0:
            plane_point = p
        elif i == 1:
            plane_point = q
        elif i == 2:
            plane_point = r
        else:
            plane_point = s
        b[2 * i, :] = plane_point[frame_num-1][0]
        b[2 * i + 1, :] = plane_point[frame_num-1][1]

    h_temp = np.matmul(a_inverse, b)
    temp = np.array([[1]])
    h_temp = np.r_[h_temp, temp]
    h = h_temp.reshape((3, 3))

    return h


def mapping(h, frame, map_object, p, q, r, s, frame_num):
    h_inverse = np.linalg.inv(h)
    points = np.array([p[frame_num - 1][0:2], q[frame_num - 1][0:2], r[frame_num - 1][0:2], s[frame_num - 1][0:2]])

    shadow = np.zeros(frame.shape[0:3], dtype=np.uint8)  # create the area size=frame size for mapping
    cv2.fillConvexPoly(shadow, points, (255, 255, 255))  # fill the mapping area with white color
    map_result = frame

    for j in range(frame.shape[0]):
        for i in range(frame.shape[1]):
            if shadow[j][i][0] == 255 and shadow[j][i][1] == 255 and shadow[j][i][2] == 255:
                point_hc = np.matmul(h_inverse, np.array([i, j, 1]))
                point = point_hc/point_hc[2]
                if 0 < point[0] < map_object.shape[1] and 0 < point[1] < map_object.shape[0]:
                    # map_result[i][k][0] = map_object[int(point[0])][int(point[1])][0]
                    # map_result[i][k][1] = map_object[int(point[0])][int(point[1])][1]
                    # map_result[i][k][2] = map_object[int(point[0])][int(point[1])][2]
                    map_result[j][i] = map_object[int(point[1])][int(point[0])]
            else:
                continue
    return map_result

