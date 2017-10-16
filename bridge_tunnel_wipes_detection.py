import numpy as np
import cv2
# import time
# import os, re
import seaborn as sns
sns.set(color_codes=True)


####__optical flow parametrs__####

old_gray_frame = 0
p0 = 0
searchin_charcts = []
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(50, 50),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))


####################################

lower_blue = np.array([70, 0, 0], dtype="uint8")  # 198, 110
upper_blue = np.array([255, 80, 20], dtype="uint8")


def grscl(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rs(img):
    return cv2.resize(img, (800, 600))


def opt_flow2(frame, frame_count):  # Lucas-Kanade
    global old_gray_frame, p0, searchin_charcts
    track_length_sum = []
    frame = rs(frame)
    mask = np.zeros_like(frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if old_gray_frame is 0:
        old_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        pass
    if p0 is 0 or p0.size == 0:
        p0 = cv2.goodFeaturesToTrack(old_gray_frame, mask=None, **feature_params)
    else:
        pass
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray_frame, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        deltax = abs(a-c)
        deltay = abs(b-d)
        if deltax + deltay >= 80:
            track_length_sum.append(deltax+deltay)

    # img = cv2.add(frame, mask)
    # cv2.imshow('frame', img)
    # cv2.imshow('p0', mask)

    if sum(track_length_sum) > 800 and len(track_length_sum) >= 8:
        searchin_charcts.append(len(track_length_sum))
        print(len(track_length_sum))
        if len(searchin_charcts) >= 2:
            searchin_charcts = []
            return 1
    # todo подумать о пропуске и обрезке кадров

    if frame_count % 70 == 0:
        old_gray_frame = 0
        p0 = 0
    else:
        old_gray_frame = frame_gray
        p0 = good_new.reshape(-1, 1, 2)
    return 0


def bridge_tunnel(img, firsts, seconds, frame_number):
    x = img.shape[1]
    y = img.shape[0]
    mask = np.zeros(img.shape[:2], np.uint8)
    mask = cv2.rectangle(mask, (int(3 * x / 8), 0),
                         (int(5 * x / 8), int(y/10)), (255, 255, 255), -1)
    # mask = cv2.rectangle(mask, (int(2 * x / 6), 0),
    #                      (int(4 * x / 6), int(y/10)), (255, 255, 255), -1)
    mask = cv2.rectangle(mask, (int(0), 0),
                         (int(1 * x / 6), int(y / 10)), (255, 255, 255), -1)
    mask = cv2.rectangle(mask, (int(5*x/6), 0),
                         (int(x), int(y / 10)), (255, 255, 255), -1)
    chunk_size = 6
    # img = rs(img)
    hist_full = cv2.calcHist([img], [0], mask, [30], [0, 30])
    nums = [hists for hists in hist_full if hists[0] > 6000]
    # print(len(nums))

    if len(firsts) < chunk_size:
        firsts.append(len(nums))
    else:
        if frame_number > chunk_size*2:
            if len(seconds) < chunk_size:
                seconds.append(firsts.pop(0))
            else:
                seconds.pop(0)
                seconds.append(firsts.pop(0))
            firsts.append(len(nums))

    if (len(firsts)) == chunk_size & len(seconds) == chunk_size:
        if abs(sum(firsts) - sum(seconds)) > 13:
            # color = ('b', 'g', 'r')
            # img = rs(img)

            return 1

    return 0
