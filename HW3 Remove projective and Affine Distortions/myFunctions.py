import numpy as np
import math as ma

# ============================ point to point correspondence method==========================
def cal_homography(distorted, undistorted):  # h: distorted->undistorted
    a = np.zeros((8, 8))  # initiate the multiplied matrix a

    for i in range(4):
        a[2 * i, :] = [distorted[i][0], distorted[i][1], 1, 0, 0, 0,
                       -distorted[i][0] * undistorted[i][0],
                       -distorted[i][1] * undistorted[i][0]]  # odd row of a
        a[2 * i + 1, :] = [0, 0, 0, distorted[i][0], distorted[i][1], 1,
                           -distorted[i][0] * undistorted[i][1],
                           -distorted[i][1] * undistorted[i][1]]  # even row of a

    a_inverse = np.linalg.inv(a)  # the inverse of a

    b = np.zeros((8, 1))
    for i in range(4):
        b[2 * i, :] = undistorted[i][0]
        b[2 * i + 1, :] = undistorted[i][1]

    h_temp = np.matmul(a_inverse, b)
    temp = np.array([[1]])
    h_temp = np.r_[h_temp, temp]
    h = h_temp.reshape((3, 3))

    return h


def mapping(h, image):
    p = np.array([0, 0, 1])
    q = np.array([0, image.shape[0], 1])
    r = np.array([image.shape[1], image.shape[0], 1])
    s = np.array([image.shape[1], 0, 1])

    new_p = np.matmul(h, p)
    new_p = new_p / new_p[2]
    new_q = np.matmul(h, q)
    new_q = new_q / new_q[2]
    new_r = np.matmul(h, r)
    new_r = new_r / new_r[2]
    new_s = np.matmul(h, s)
    new_s = new_s / new_s[2]

    x_min = int(min(new_p[0], new_q[0], new_r[0], new_s[0]))
    x_max = int(max(new_p[0], new_q[0], new_r[0], new_s[0]))
    y_min = int(min(new_p[1], new_q[1], new_r[1], new_s[1]))
    y_max = int(max(new_p[1], new_q[1], new_r[1], new_s[1]))

    x_length = x_max - x_min
    y_length = y_max - y_min

    shadow = np.zeros([y_length, x_length, 3], dtype=np.uint8)  # create the area size for mapping

    h_inverse = np.linalg.inv(h)

    for j in range(y_length):
        for i in range(x_length):
            temp_point = np.matmul(h_inverse, np.array([i + x_min, j + y_min, 1]))
            point = temp_point / temp_point[2]
            if 0 < point[0] < image.shape[1] and 0 < point[1] < image.shape[0]:
                shadow[j][i] = image[int(point[1])][int(point[0])]

    return shadow


# =====================================================================================
# =================================two-step methods====================================
def v_line(p, q, r, s):
    line_pq = np.cross(p, q)
    line_rs = np.cross(r, s)
    v_point_pq_rs = np.cross(line_pq, line_rs)  # vanishing point of pq&rs
    v_point_pq_rs = v_point_pq_rs / v_point_pq_rs[2]
    line_ps = np.cross(p, s)
    line_qr = np.cross(q, r)
    v_point_ps_qr = np.cross(line_ps, line_qr)  # vanishing point of ps&qr
    v_point_ps_qr = v_point_ps_qr / v_point_ps_qr[2]

    vanishing_line = np.cross(v_point_pq_rs, v_point_ps_qr)  # vanishing line
    return vanishing_line / vanishing_line[2]


def homography_vline_back(line):
    h = np.zeros((3, 3))
    h[0][0] = 1
    h[1][1] = 1
    h[2] = line
    return h


def homography_affine(h_vl, img_coor):
    new_p = np.matmul(h_vl, img_coor[0])
    new_p = new_p / new_p[2]
    new_q = np.matmul(h_vl, img_coor[1])
    new_q = new_q / new_q[2]
    new_r = np.matmul(h_vl, img_coor[2])
    new_r = new_r / new_r[2]
    new_s = np.matmul(h_vl, img_coor[3])
    new_s = new_s / new_s[2]

    new_l1 = np.cross(new_p, new_q)
    new_l1 = new_l1 / new_l1[2]
    new_m1 = np.cross(new_s, new_p)
    new_m1 = new_m1 / new_m1[2]
    new_l2 = np.cross(new_r, new_s)
    new_l2 = new_l2 / new_l2[2]
    new_m2 = np.cross(new_q, new_r)
    new_m2 = new_m2 / new_m2[2]

    a = np.zeros((2, 2))
    a[0] = [new_l1[0] * new_m1[0], new_l1[0] * new_m1[1] + new_l1[1] * new_m1[0]]
    a[1] = [new_l2[0] * new_m2[0], new_l2[0] * new_m2[1] + new_l2[1] * new_m2[0]]
    b = np.zeros((2, 1))
    b[0] = -new_l1[1] * new_m1[1]
    b[1] = -new_l2[1] * new_m2[1]
    x = np.matmul(np.linalg.inv(a), b)

    s = np.ones((2, 2))
    s[0] = np.transpose(x)
    s[1][0] = x[1]

    u, d_2, v = np.linalg.svd(s)
    d = np.sqrt(d_2)

    D = np.diag(d)
    A = np.matmul(np.matmul(u, D), np.transpose(u))

    h = np.zeros((3, 3))
    h[0, :2] = A[0]
    h[1, :2] = A[1]
    h[2][2] = 1

    return np.linalg.inv(h)  # affine -> normal


# ======================one-step method============================================

def onestep(set1, set2):
    # finding five sets of orthogonal lines l_i v.s. m_i
    l1 = np.cross(set1[0], set1[1])  # set1 pq
    l1 = l1 / l1[2]
    m1 = np.cross(set1[0], set1[3])  # set1 ps
    m1 = m1 / m1[2]

    l2 = np.cross(set1[1], set1[2])  # set1 qr
    l2 = l2 / l2[2]
    m2 = np.cross(set1[2], set1[3])  # set1 rs
    m2 = m2 / m2[2]

    l3 = np.cross(set1[0], set1[3])  # set1 ps
    l3 = l3 / l3[2]
    m3 = np.cross(set1[2], set1[3])  # set1 rs
    m3 = m3 / m3[2]

    l4 = np.cross(set2[0], set2[1])  # set2 pq
    l4 = l4 / l4[2]
    m4 = np.cross(set2[1], set2[2])  # set2 qr
    m4 = m4 / m4[2]

    l5 = np.cross(set2[0], set2[1])  # set2 pq
    l5 = l5 / l5[2]
    m5 = np.cross(set2[0], set2[3])  # set2 ps
    m5 = m5 / m5[2]

    l = np.array([l1, l2, l3, l4, l5])
    m = np.array([m1, m2, m3, m4, m5])

    a = np.zeros((5, 5))
    b = np.zeros((5, 1))
    for i in range(5):
        a[i] = [l[i][0] * m[i][0], l[i][0] * m[i][1] + l[i][1] * m[i][0], l[i][1] * m[i][1],
                l[i][0] * m[i][2] + l[i][2] * m[i][0], l[i][1] * m[i][2] + l[i][2] * m[i][1]]
        b[i] = -l[i][2] * m[i][2]

    x = np.matmul(np.linalg.inv(a), b)
    x = x / max(x)  # x = [a \\ b/2 \\ c \\ d/2 \\ e/2 \\ f]

    s = np.zeros((2, 2))
    s[0][0] = x[0]
    s[0][1] = x[1]
    s[1][0] = x[1]
    s[1][1] = x[2]

    u, d_2, ut = np.linalg.svd(s)  # calculate SVD of s

    d = np.sqrt(d_2)
    D = np.diag(d)
    A = np.matmul(np.matmul(u, D), np.transpose(u))

    temp = np.array([x[3], x[4]])
    v = np.matmul(np.linalg.inv(A), temp)  # calculate v in h

    h = np.zeros((3, 3))
    h[0, :2] = A[0]
    h[1, :2] = A[1]
    h[2][2] = 1
    h[2][0] = v[0]
    h[2][1] = v[1]

    return h


# ==============================extra credit===============================
def estimate_ratio(h_vl, img_coor):
    new_p = np.matmul(h_vl, img_coor[0])
    new_p = new_p / new_p[2]
    new_q = np.matmul(h_vl, img_coor[1])
    new_q = new_q / new_q[2]
    new_r = np.matmul(h_vl, img_coor[2])
    new_r = new_r / new_r[2]
    new_s = np.matmul(h_vl, img_coor[3])
    new_s = new_s / new_s[2]

    new_l1 = np.cross(new_p, new_q)
    new_l1 = new_l1 / new_l1[2]
    new_m1 = np.cross(new_s, new_p)
    new_m1 = new_m1 / new_m1[2]
    new_l2 = np.cross(new_r, new_s)
    new_l2 = new_l2 / new_l2[2]
    new_m2 = np.cross(new_q, new_r)
    new_m2 = new_m2 / new_m2[2]

    a = np.zeros((2, 2))
    a[0] = [new_l1[0] * new_m1[0], new_l1[0] * new_m1[1] + new_l1[1] * new_m1[0]]
    a[1] = [new_l2[0] * new_m2[0], new_l2[0] * new_m2[1] + new_l2[1] * new_m2[0]]
    b = np.zeros((2, 1))
    b[0] = -new_l1[1] * new_m1[1]
    b[1] = -new_l2[1] * new_m2[1]
    x = np.matmul(np.linalg.inv(a), b)

    s = np.ones((2, 2))
    s[0] = np.transpose(x)
    s[1][0] = x[1]

    line_pq = np.cross(img_coor[0], img_coor[1])  # use qpr as alpha
    line_pq = line_pq / line_pq[2]
    line_pr = np.cross(img_coor[0], img_coor[2])
    line_pr = line_pr / line_pr[2]
    line_qr = np.cross(img_coor[1], img_coor[2])  # use qrp as beta
    line_qr = line_qr / line_qr[2]

    numer1 = (np.matmul(np.matmul(line_pq[:2], s), np.transpose(line_pr[:2])))**2
    denom1 = np.matmul(np.matmul(line_pq[:2], s), np.transpose(line_pq[:2])) * np.matmul(np.matmul(line_pr[:2], s),
                                                                                         np.transpose(line_pr[:2]))
    cos_alpha_2 = numer1 / denom1
    sin_alpha = ma.sqrt(1 - cos_alpha_2)

    numer2 = np.square(np.matmul(np.matmul(line_qr[:2], s), np.transpose(line_pr[:2])))
    denom2 = np.matmul(np.matmul(line_qr[:2], s), np.transpose(line_qr[:2])) * np.matmul(np.matmul(line_pr[:2], s),
                                                                                         np.transpose(line_pr[:2]))
    cos_beta_2 = numer2 / denom2
    sin_beta = ma.sqrt(1 - cos_beta_2)

    return sin_alpha / sin_beta
