import cv2


def sift_features(img):
    '''

    :param img: original colored picture
    :return: colored picture with interest points & keypoint & des
    '''
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray_img, None)  # kp will be a list of keypoints and des is a numpy array of shape
    img = cv2.drawKeypoints(gray_img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img, kp, des


def sift_match(img1, img2, max_show=200, mode='bruteForce'):
    img1, kp1, des1 = sift_features(img1)
    img2, kp2, des2 = sift_features(img2)

    if mode == 'bruteForce':
        # Creating BFMatcher object
        my_bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Getting matches between both descriptors
        matches = my_bf.match(des1, des2)

        # Sorting matches according to the distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Drawing N matches
        if len(matches) > max_show:
            img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_show],
                                  flags=2, outImg=None)
        else:
            img = cv2.drawMatches(img1, kp1, img2, kp2, matches,
                                  flags=2, outImg=None)
    elif mode == 'knn':
        my_bf = cv2.BFMatcher()
        matches = my_bf.knnMatch(des1, des2, k=2)

        # Taking match only if the distance's proportion is met
        best_one = []
        for x, y in matches:
            if x.distance < 0.7 * y.distance:
                best_one.append([x])

        # Drawing N matches
        img = cv2.drawMatchesKnn(img1, kp1, img2, kp2,
                                 best_one[:max_show], flags=2, outImg=None)

    return img
