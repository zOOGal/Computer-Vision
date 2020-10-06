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

frame_1 = cv2.imread('painting1.jpeg')
frame_2 = cv2.imread('painting2.jpeg')
frame_3 = cv2.imread('painting3.jpeg')

kitten = cv2.imread('kittens.jpeg')

P = list()
P.append([234, 414, 1])  # P's coordinates in frame 1
P.append([180, 525, 1])  # P's coordinates in frame 2
P.append([90, 254, 1])  # P's coordinates in frame 3

S = list()
S.append([1904, 200, 1])  # S's coordinates in frame 1
S.append([1950, 640, 1])  # S's coordinates in frame 2
S.append([1356, 117, 1])  # S's coordinates in frame 3

Q = list()
Q.append([145, 1675, 1])  # Q's coordinates in frame 1
Q.append([181, 2493, 1])  # Q's coordinates in frame 2
Q.append([70, 1414, 1])  # Q's coordinates in frame 3

R = list()
R.append([1802, 1970, 1])  # R's coordinates in frame 1
R.append([1955, 2080, 1])  # R's coordinates in frame 2
R.append([1200, 2040, 1])  # R's coordinates in frame 3

cor_kittens = list()
cor_kittens.append([0, 0, 1])  # P's coordinates in object "kittens"
cor_kittens.append([0, 1125, 1])  # Q's coordinates in object "kittens"
cor_kittens.append([1920, 1125, 1])  # R's coordinates in object "kittens"
cor_kittens.append([1920, 0, 1])  # S's coordinates in object "kittens"
#pdb.set_trace()

h_1 = funcForHW2.cal_homography(P, Q, R, S, 1, cor_kittens)
result_1 = funcForHW2.mapping(h_1, frame_1, kitten, P, Q, R, S, 1)
result_1 = cv2.cvtColor(result_1, cv2.COLOR_BGR2RGB)
pixels_1 = np.array(result_1)
plt.imshow(pixels_1)
plt.show()

h_2 = funcForHW2.cal_homography(P, Q, R, S, 2, cor_kittens)
result_2 = funcForHW2.mapping(h_2, frame_2, kitten, P, Q, R, S, 2)
result_2 = cv2.cvtColor(result_2, cv2.COLOR_BGR2RGB)
pixels_2 = np.array(result_2)
plt.imshow(pixels_2)
plt.show()

h_3 = funcForHW2.cal_homography(P, Q, R, S, 3, cor_kittens)
result_3 = funcForHW2.mapping(h_3, frame_3, kitten, P, Q, R, S, 3)
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
result_4 = funcForHW2.mapping(h_ac, frame_3, frame_1, P, Q, R, S, 3)
result_4 = cv2.cvtColor(result_4, cv2.COLOR_BGR2RGB)
pixels_4 = np.array(result_4)
plt.imshow(pixels_4)
plt.show()
pdb.set_trace()