import numpy as np
import cv2
import time
import os, re
from matplotlib import pyplot as plt
from pylab import *

# constants
text_file = 'C:/Users/janch/PycharmProjects/orange/output_val1.txt'
start_time = time.time()
directory = 'D:/visionhack/validationset/'
r = re.compile(".*avi")
files = filter(r.match, [f for f in os.listdir(directory)])

lower_blue = np.array([70, 0, 0], dtype="uint8")  # 198, 110
upper_blue = np.array([255, 70, 20], dtype="uint8")


def grscl(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rs(img):
    return cv2.resize(img, (800, 600))


def persp_transformation(img):
    x = img.shape[1]
    y = img.shape[0]
    img_size = (x, y)
    source_points = np.float32([
    [0.117 * x, y],
    [(0.5 * x) - (x*0.078), (2/3)*y],
    [(0.5 * x) + (x*0.078), (2/3)*y],
    [x - (0.117 * x), y]
    ])

    destination_points = np.float32([
        [0.25 * x, y],
        [0.25 * x, 0],
        [x - (0.25 * x), 0],
        [x - (0.25 * x), y]
    ])

    persp_transform = cv2.getPerspectiveTransform(source_points, destination_points)
    inverse_persp_transform = cv2. getPerspectiveTransform(destination_points, source_points)

    warped_img = cv2.warpPerspective(img, persp_transform, img_size, flags=cv2.INTER_LINEAR)  # ???

    return warped_img, inverse_persp_transform  # warped_img is the result we need


def temp_matchin(img):
    # plt.ion()
    img2 = img.copy()
    template = cv2.imread('C:/Users/janch/Desktop/vh_templates/created/walkin_man.jpg', 1)
    # w, h = template.shape[::-1]
    h = template.shape[0]
    w = template.shape[1]
    img = img2.copy()
    method = cv2.TM_CCOEFF
    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    # plt.subplot(121), plt.imshow(res, cmap='gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(img, cmap='gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(method)
    cv2.imshow('rected', rs(img))
    # plt.show()
    # plt.close(1)


def epsareas(blue_contours, img):
    black = np.zeros([img.shape[:2][0], img.shape[:2][1], 1], dtype="uint8")


def histogram(img, firsts, seconds, frame_number):
    # ion()
    x = img.shape[1]
    y = img.shape[0]
    mask = np.zeros(img.shape[:2], np.uint8)
    mask = cv2.rectangle(mask, (int(x / 4), 0),
                         (int(3 * x / 4), int(y/8)), (255, 255, 255), -1)
    # mask[int(100):300, # y len
    # int(100): 300] = 255
    # cv2.imshow('mask', rs(mask))
    chunk_size = 5
    # img = rs(img)
    hist_full = cv2.calcHist([img], [0], mask, [256], [0, 256])
    nums = [hists for hists in hist_full if hists[0] > 1000]
    print(len(nums))

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

    # if (frame_number > chunk_size * 2) & (len(seconds) < chunk_size):
    #     seconds.append(len(nums))
    # else:
    #     if frame_number > chunk_size * 2:
    #         seconds.pop(0)
    #         seconds.append(len(nums))

    if (len(firsts)) == chunk_size & len(seconds) == chunk_size:
        if abs(sum(firsts) - sum(seconds)) > 100:
            print('trololololo')
            return 1

    # for hists in hist_full:
    #     enumerate([hists[0] > 1000])
    # enumerate(hist_full> 1000)
    # plot(hist_full)
    # cv2.imshow('hist', hist_full)
    # ion()
    # color = ('b', 'g', 'r')
    # img = rs(img)
    # for i, col in enumerate(color):
    #     histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    #
    #     plt.xlim([0, 256])
    # plot(histr, color=col)
    # draw()
    # plt.show()


def blue_mask(img):
    kernel = np.ones((80,80),np.uint8)

    mask_blue = cv2.inRange(cv2.medianBlur(img, 5), lower_blue, upper_blue)

    ret, thresh = cv2.threshold(mask_blue, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', rs(thresh))
    threshed =cv2.dilate(thresh, kernel, iterations=1)


    masked = cv2.bitwise_and(img, img, mask=threshed)
    # _, contours_blue, hierarchy = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('blue', rs(masked))
    # corners
    edges = cv2.Canny(grscl(masked), 0, 255)
    cv2.imshow('canny', rs(edges))
    # blue_areas = [cv2.boundingRect(cbl) for cbl in contours_blue]
    return masked


def processing(capture):
    frame_skip = 1
    frame_count = capture.get(cv2.CAP_PROP_POS_FRAMES)
    firsts = []
    seconds = []
    while True:
        try:
            if frame_count % frame_skip == 0:
                    ret, frame = capture.read()
            # cv2.imshow('transformed', rs(persp_transformation(frame)[0]))
            # cv2.imshow('frame', rs(frame))
            if histogram(frame, firsts, seconds, frame_count) == 1:
                print(file)
                return '100000'
                # break
            # plt.close()
            # plt.ion()
            # blue_mask(frame)
            # temp_matchin(blue_mask(frame))
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(-1)
                break
        except:
            return '000000'


for file in files:

    filepath = (directory + file)
    cap = cv2.VideoCapture(filepath)
    start_ret, start_frame = cap.read()

    out = processing(cap)
    with open(text_file, 'a') as text:
        text.write("%s %s \n" % (file, out))

    cap.release()
    cv2.destroyAllWindows()