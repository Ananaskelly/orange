import numpy as np
import cv2
import time
import os, re
# from matplotlib import pyplot as plt
# from pylab import *

# globals()
bridge = 0
bump = 0
text_file = 'C:/Users/janch/PycharmProjects/orange/output_val12.txt'
start_time = time.time()
directory = 'D:/visionhack/validationset/'
r = re.compile(".*avi")
files = filter(r.match, [f for f in os.listdir(directory)])

lower_blue = np.array([70, 0, 0], dtype="uint8")  # 198, 110
upper_blue = np.array([255, 80, 20], dtype="uint8")


def grscl(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rs(img):
    return cv2.resize(img, (800, 600))


def h_cascade(img, orig, cascade='C:/Users/janch/Desktop/vh_templates/HAAR/haarcascade/cascade.xml'):
    # blue_mask(img)
    # epsarea = [130,130]
    area = [100**2, 130**2]
    detector = cv2.CascadeClassifier(cascade)
    rects = detector.detectMultiScale(img, scaleFactor=1.3,
                                      minNeighbors=10, minSize=(10, 10))
    # if len(rects) > 0:
    #     return 1
    for (i, (x, y, w, h)) in enumerate(rects):
        if area[0] <= w*h <= area[1]:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
            return 1
    return 0
            # if img.shape[1] - x+w < 300:
            #     print('tololololland')
    # cv2.imshow("sign", rs(img))


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


# def temp_matchin(img):
#     # plt.ion()
#     img2 = img.copy()
#     template = cv2.imread('C:/Users/janch/Desktop/vh_templates/created/walkin_man.jpg', 1)
#     # w, h = template.shape[::-1]
#     h = template.shape[0]
#     w = template.shape[1]
#     img = img2.copy()
#     method = cv2.TM_CCOEFF
#     # Apply template Matching
#     res = cv2.matchTemplate(img, template, method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#
#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#
#     cv2.rectangle(img, top_left, bottom_right, 255, 2)
#
#     # plt.subplot(121), plt.imshow(res, cmap='gray')
#     # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     # plt.subplot(122), plt.imshow(img, cmap='gray')
#     # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     # plt.suptitle(method)
#     cv2.imshow('rected', rs(img))
#     # plt.show()
#     # plt.close(1)


# def epsareas(blue_contours, img):
#     black = np.zeros([img.shape[:2][0], img.shape[:2][1], 1], dtype="uint8")


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

    return 0
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
    black = np.zeros([img.shape[:2][0], img.shape[:2][1], 1], dtype="uint8")
    kernel = np.ones((60, 60), np.uint8)
    try:
        mask_blue = cv2.inRange(img, lower_blue, upper_blue)
        # mask_blue = cv2.inRange(cv2.medianBlur(img, 5), lower_blue, upper_blue)
    except:
        return 0
    thresh = mask_blue
    # ret, thresh = cv2.threshold(mask_blue, 127, 255, cv2.THRESH_BINARY)
    threshed = cv2.dilate(thresh, kernel, iterations=1)
    im2, contours, hierarchy = cv2.findContours(threshed, 1, 2)
    areas_space = [cv2.contourArea(c) for c in contours]
    ind_big_spaces = [i for i, x in enumerate(areas_space) if x > 30]  # площадь максимально маленькой области
    bareas = [contours[ind] for ind in ind_big_spaces]

    for cnt in bareas:
        x,y,w,h = cv2.boundingRect(cnt)
        # rect = img[x: x + w, y: y + h]
        black = cv2.rectangle(black, (x, y),
                                     (x + w , y + h), (255, 255, 255), -1)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    masked = cv2.bitwise_and(img, img, mask=black)
    # cv2.imshow('rected', rs(masked))

    # masked = cv2.bitwise_and(img, img, mask=threshed)


    # if len(mask_blue_mass) < 3:  # задержка в кадрах перед исчезновением красной области
    #     mask_blue_mass.append(masked)
    # else:
    #     mask_blue_mass.pop(0)
    #     mask_blue_mass.append(masked)
    #
    # sum_rmask = masked
    # if len(mask_blue_mass) > 0:
    #     for bmask in mask_blue_mass:
    #         sum_rmask = cv2.addWeighted(bmask, 0.5, sum_rmask, 1, 1)
    # cv2.imshow('summ_rmask', rs(sum_rmask))


    # _, contours_blue, hierarchy = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('blue', rs(masked))
    # corners
    # edges = cv2.Canny(grscl(masked), 0, 255)
    # cv2.imshow('canny', rs(edges))
    # blue_areas = [cv2.boundingRect(cbl) for cbl in contours_blue]
    return h_cascade(masked, img)


def processing(capture):
    # globals()
    bridge = 0
    bump =0
    frame_skip = 1
    frame_count = capture.get(cv2.CAP_PROP_POS_FRAMES)
    firsts = []
    seconds = []
    while True:
        try:
            if frame_count % frame_skip == 0:
                    ret, frame = capture.read()
                    if bump == 0:
                        if blue_mask(frame) == 1:
                            bump = 1
                    if bridge == 0:
                        if histogram(frame, firsts, seconds, frame_count) == 1:
                            bridge = 1
                    cv2.imshow('frame', rs(frame))
            frame_count += 1
        except:
            print('first_try')
            return '%s0000%s' % (bridge, bump)

        # if frame_count >= 298:
        #     break
            #####################

            # print(file)
            # return '100000'
        # else:
        #     return '000000'
            ##########################
            # break
            # cv2.imshow('transformed', rs(persp_transformation(frame)[0]))
            # cv2.imshow('frame', rs(frame))

            # plt.close()
            # plt.ion()
            # blue_mask(frame)
            # temp_matchin(blue_mask(frame))
        # h_cascade(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(-1)
            break
        # return '%s0000%s' % (bridge, zebra)
    return '%s0000%s' % (bridge, bump)


for file in files:
    globals()
    # bridge = 0
    # bump = 0
    filepath = (directory + file)
    cap = cv2.VideoCapture(filepath)
    start_ret, start_frame = cap.read()
    # mask_blue_mass = []

    out = processing(cap)
    with open(text_file, 'a') as text:
        text.write("%s %s \n" % (file, out))

    cap.release()
    cv2.destroyAllWindows()