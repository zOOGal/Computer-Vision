# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 23:38:12 2020

@author: Xu Gao

HW2 code for ece661 computer vision
"""

import cv2 as cv2
import pdb
import sys

sys.path.append('E:\clare\lectureMaterials\661\hws')
import funcForHW2
import matplotlib.pyplot as plt
import numpy as np

bk1 = cv2.imread('bk1.jpg')
bk2 = cv2.imread('bk2.jpg')
bk3 = cv2.imread('bk3.jpg')

poster = cv2.imread('shameless.jpg')

P = list()
P.append([309, 238, 1])  # P's coordinates in frame 1
P.append([329, 244, 1])  # P's coordinates in frame 2
P.append([297, 250, 1])  # P's coordinates in frame 3

S = list()
S.append([716, 206, 1])  # S's coordinates in frame 1
S.append([741, 119, 1])  # S's coordinates in frame 2
S.append([907, 263, 1])  # S's coordinates in frame 3

Q = list()
Q.append([256, 803, 1])  # Q's coordinates in frame 1
Q.append([268, 924, 1])  # Q's coordinates in frame 2
Q.append([163, 848, 1])  # Q's coordinates in frame 3

R = list()
R.append([674, 833, 1])  # R's coordinates in frame 1
R.append([687, 993, 1])  # R's coordinates in frame 2
R.append([926, 884, 1])  # R's coordinates in frame 3

coor_poster = list()
coor_poster.append([0, 0, 1])  # P's coordinates in object "poster"
coor_poster.append([0, 1000, 1])  # Q's coordinates in object "poster"
coor_poster.append([1000, 1000, 1])  # R's coordinates in object "poster"
coor_poster.append([1000, 0, 1])  # S's coordinates in object "poster"

h_1 = funcForHW2.cal_homography(P, Q, R, S, 1, coor_poster)
result_1 = funcForHW2.mapping(h_1, bk1, poster, P, Q, R, S, 1)
result_1 = cv2.cvtColor(result_1, cv2.COLOR_BGR2RGB)
pixels_1 = np.array(result_1)
plt.imshow(pixels_1)
plt.show()

h_2 = funcForHW2.cal_homography(P, Q, R, S, 2, coor_poster)
result_2 = funcForHW2.mapping(h_2, bk2, poster, P, Q, R, S, 2)
result_2 = cv2.cvtColor(result_2, cv2.COLOR_BGR2RGB)
pixels_2 = np.array(result_2)
plt.imshow(pixels_2)
plt.show()

h_3 = funcForHW2.cal_homography(P, Q, R, S, 3, coor_poster)
result_3 = funcForHW2.mapping(h_3, bk3, poster, P, Q, R, S, 3)
result_3 = cv2.cvtColor(result_3, cv2.COLOR_BGR2RGB)
pixels_3 = np.array(result_3)
plt.imshow(pixels_3)
plt.show()


coor_1 = list()
coor_1.append(P[0])  # P's coordinates in object
coor_1.append(Q[0])  # Q's coordinates in object
coor_1.append(R[0])  # R's coordinates in object
coor_1.append(S[0])  # S's coordinates in object
h_ab = funcForHW2.cal_homography(P, Q, R, S, 2, coor_1)

coor_2 = list()
coor_2.append(P[1])  # P's coordinates in object
coor_2.append(Q[1])  # Q's coordinates in object
coor_2.append(R[1])  # R's coordinates in object
coor_2.append(S[1])  # S's coordinates in object
h_bc = funcForHW2.cal_homography(P, Q, R, S, 3, coor_2)

h_ac = np.matmul(h_ab, h_bc)
result_4 = funcForHW2.mapping(h_ac, bk3, bk1, P, Q, R, S, 3)
result_4 = cv2.cvtColor(result_4, cv2.COLOR_BGR2RGB)
pixels_4 = np.array(result_4)
plt.imshow(pixels_4)
plt.show()

pdb.set_trace()
