import numpy as np
import cv2
import time
import os, re
import seaborn as sns
from pylab import *
sns.set(color_codes=True)

# globals()
bridge = 0
bump = 0
count = 1
text_file = 'C:/Users/janch/PycharmProjects/orange/output_wipes/output5.txt'
start_time = time.time()
# directory = 'D:/visionhack/validationset/'
directory = 'D:/visionhack/testset/'
r = re.compile(".*avi")
files = filter(r.match, [f for f in os.listdir(directory)])
# files = enumerate(files)

####optical flow parametrs####################################################

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


##############################################################################

lower_blue = np.array([70, 0, 0], dtype="uint8")  # 198, 110
upper_blue = np.array([255, 80, 20], dtype="uint8")


def grscl(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rs(img):
    return cv2.resize(img, (800, 600))


'''def h_cascade(img, orig, cascade='C:/Users/janch/Desktop/vh_templates/HAAR/haarcascade/cascade.xml'):
    # blue_mask(img)
    # epsarea = [130,130]
    area = [90**2, 130**2]
    detector = cv2.CascadeClassifier(cascade)
    rects = detector.detectMultiScale(img, scaleFactor=1.3,
                                      minNeighbors=10, minSize=(10, 10))
    if len(rects) > 0:
        x, y, w, h = rects[0]
        if area[0] <= w * h <= area[1]:
            cv2.imwrite('trainset/bump/%s.jpg' % file, orig[y:y + h, x:x + w])
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
            return 1
    # for (i, (x, y, w, h)) in enumerate(rects):
    #     if area[0] <= w*h <= area[1]:
    #         cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #         cv2.imwrite('trainset/%s' % file, orig[x:x+w, y:y+h])
    #         return 1
    # return 0
            # if img.shape[1] - x+w < 300:
            #     print('tololololland')
    # cv2.imshow("sign", rs(img))
'''

'''def persp_transformation(img):
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
'''


'''def phase_correlate(img, srcs):

    img = cv2.cvtColor(rs(img), cv2.COLOR_BGR2GRAY)
    center = tuple([int(img.shape[1]/2), int(img.shape[0]/2)])
    # center = ()
    src = np.float32(img)
    if len(srcs) < 2:
        srcs.append(src)
    else:
        srcs.pop()
        srcs.append(src)
        shift = cv2.phaseCorrelate(srcs[0], srcs[1])
        radius = sqrt(shift[0][1]*shift[0][1]+shift[0][0]*shift[0][0])
        if radius > 0:
            cv2.circle(img, center, 20, (0,255,0), 10)
            cv2.line(img, center, (int(center[0]+shift[0][0]), int(center[1]+shift[0][1])), (0,255,0), 10)
        cv2.imshow('corelated', rs(img))'''


def opt_flow2(frame):  # Lucas-Kanade
    global old_gray_frame, p0, searchin_charcts
    track_length_sum = []
    frame = rs(frame)
    mask = np.zeros_like(frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if old_gray_frame is 0:
        old_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        pass
    if p0 is 0:
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

    if sum(track_length_sum) > 750 and len(track_length_sum) >= 7:
        searchin_charcts.append(len(track_length_sum))
        # print(len(track_length_sum))
        if len(searchin_charcts) >= 3:
            return 1
    # todo подумать о пропуске и обрезке кадров
    img = cv2.add(frame, mask)
    # cv2.imshow('frame', img)
    # cv2.imshow('p0', mask)
    old_gray_frame = frame_gray
    p0 = good_new.reshape(-1, 1, 2)
    return 0


'''def opt_flow1(frame2, frame1):
    frame1 = rs(frame1)
    frame2 = rs(frame2)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.9, 3, 15, 3, 20, 2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2', bgr)
    prvs = next'''


def bridge_tunnel(img, firsts, seconds, frame_number):
    # ion()
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
    # mask[int(100):300, # y len
    # int(100): 300] = 255
    # cv2.imshow('mask', rs(mask))
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

    # if (frame_number > chunk_size * 2) & (len(seconds) < chunk_size):
    #     seconds.append(len(nums))
    # else:
    #     if frame_number > chunk_size * 2:
    #         seconds.pop(0)
    #         seconds.append(len(nums))

    if (len(firsts)) == chunk_size & len(seconds) == chunk_size:
        if abs(sum(firsts) - sum(seconds)) > 13:
            # cv2.imshow('orig', rs(img))
            # print('trololololo')
            # for hists in hist_full:
            #     enumerate([hists[0] > 1000])
            # enumerate(hist_full> 1000)
            # plot(hist_full)
            # cv2.imshow('hist', hist_full)
            # ion()
            color = ('b', 'g', 'r')
            img = rs(img)
            for i, col in enumerate(color):
                histr = cv2.calcHist([img], [i], None, [256], [0, 256])

                # plt.xlim([0, 256])
            # plot(histr, color=col)
            # draw()
            # plt.show()
            return 1

    return 0


'''def blue_mask(img):
    black = np.zeros([img.shape[:2][0], img.shape[:2][1], 1], dtype="uint8")
    kernel = np.ones((50,50), np.uint8)
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
    # return (h_cascade(masked, img),
    #         h_cascade(masked, img, cascade='C:/Users/janch/Desktop/vh_templates/HAAR/haarcascade_cross/cascade.xml'))
    return h_cascade(masked, img)
'''


def processing(capture):
    frame_count = capture.get(cv2.CAP_PROP_POS_FRAMES)
    global p0, old_gray_frame, searchin_charcts
    old_gray_frame = 0
    p0 = 0
    searchin_charcts = []
    ##########################
    zebra = 0
    bridge = 0
    bump = 0
    frame_skip = 1
    wipe = 0

    firsts = []
    seconds = []
    while True:
        try:
            if frame_count % frame_skip == 0:
                    ret, frame = capture.read()
                    if ret:
                        pass
                    else:
                        return '%s00%s%s%s' % (bridge, bump, wipe, zebra)
                    if opt_flow2(frame) == 1:
                        wipe = 1
                    if frame_count % 70 == 0:
                        old_gray_frame = 0
                        p0 = 0

                    # frame = rs(frame)
                    # if bump == 0 or zebra == 0:
                    #     bump, zebra = blue_mask(frame)
                    # BUUUMP
                    # if bump == 0:
                    #     if blue_mask(frame) == 1:
                    #         bump = 1

                    if bridge == 0:
                        if bridge_tunnel(frame, firsts, seconds, frame_count) == 1:
                            bridge = 1
                    # cv2.imshow('frame', rs(frame))
                    # plt.pause(0.05)
            frame_count += 1
        except:
            # print('first_try')
            return '%s00%s%s%s' % (bridge, bump, wipe, zebra)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(-1)
            break
        # return '%s0000%s' % (bridge, zebra)
    return '%s00%s%s%s' % (bridge, bump,wipe, zebra)


for file in files:
    filepath = (directory + file)
    cap = cv2.VideoCapture(filepath)
    start_ret, start_frame = cap.read()
    # mask_blue_mass = []

    out = processing(cap)
    with open(text_file, 'a') as text:
        text.write("%s %s \n" % (file, out))
    print('processed № %s file %s  ' % (count, file))
    count += 1

    cap.release()
    cv2.destroyAllWindows()