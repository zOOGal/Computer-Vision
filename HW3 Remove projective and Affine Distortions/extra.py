import myFunctions
import numpy as np

img = list()
img.append([131, 59, 1])
img.append([178, 790, 1])
img.append([483, 717, 1])
img.append([459, 126, 1])

vl = myFunctions.v_line(img[0], img[1], img[2], img[3])
h_vl = myFunctions.homography_vline_back(vl)
try:
    ratio = myFunctions.estimate_ratio(h_vl, img)
except (np.linalg.LinAlgError, ValueError):
    ratio = 0
    print('Choose another set please!')
if ratio:
    print(ratio)
