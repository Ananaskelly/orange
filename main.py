import numpy as np
import cv2
import time
import os, re
import matplotlib as plt

start_time = time.time()
directory = 'C:/Users/janch/Desktop/trainset/'
r = re.compile(".*avi")
files = filter(r.match, [f for f in os.listdir(directory)])


def rs1(img):
    return cv2.resize(img, (800, 600))

def rs(img):
    return cv2.resize(img, (320, 240))


def zebra(img):
    kernel1 = np.ones((5, 5), np.uint8)
    lower_w = (150)
    upper_w = (240)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    kernel = (5, 5)
    # img = cv2.GaussianBlur(img, kernel, 0)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # clahed = gray_img
    clahed = clahe.apply(gray_img)
    clahed = cv2.GaussianBlur(clahed, kernel, 0)
    cv2.imshow('gray', rs(gray_img))
    gray_img = persp_transformation(clahed)[0]  # gray_img
    cv2.imshow('clahed_transform', rs(gray_img))

    # base thresh
    th = cv2.inRange(gray_img, lower_w, upper_w)
    # reth, th1 = cv2.threshold(gray_img, lower_w, upper_w, cv2.THRESH_OTSU)

    # erosed = cv2.erode(th1, kernel, iterations=10)
    # cv2.imshow('OTSU erode', rs(erosed))
    # cv2.imshow('OTSU opening', rs(closing))

    th2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel1)

    # hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # l = hls_img[:, :, 1]
    # s = hls_img[:, :, 2]
    # binary_output = np.zeros_like(s)
    # binary_output[(s >= sthresh[0]) & (s <= sthresh[1])
    #               & (l >= lthresh[0]) & (l <= lthresh[1])] = 255
    cv2.imshow('mean_thrsh', rs(th2))
    cv2.imshow('base thrashold', rs(th))
    cv2.imshow('adapt open', rs1(th2))
    # cv2.imshow('clahed', rs(clahed))
    # cv2.imshow('thrashed2', rs(th2))
    # cv2.imshow('bin', rs(binary_output))
    # cv2.imshow('frame', rs(img))
    # return binary_output


def persp_transformation(img):
    x = img.shape[1]
    y = img.shape[0]
    img_size = (x, y)
    source_points = np.float32([
        [0.217 * x, y],
        [(0.5 * x) - (x * 0.068), (3 / 4) * y],
        [(0.5 * x) + (x * 0.068), (3 / 4) * y],
        [x - (0.217 * x), y]
    ])
    # [0.117 * x, y],
    # [(0.5 * x) - (x*0.078), (2/3)*y],
    # [(0.5 * x) + (x*0.078), (2/3)*y],
    # [x - (0.117 * x), y]
    # ])

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


def processing(capture):
    frame_skip = 1
    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        try:
            if frame_count % frame_skip == 0:
                ret, frame = cap.read()
                zebra(frame)
                cv2.imshow('orig', rs(frame))
            # cv2.imshow('transformed', rs(persp_transformation(frame)[0]))
            # cv2.imshow('frame', rs(frame))
            if cv2.waitKey(30) & 0xFF == ord('q'):
                # print(-1)
                break
        except:
            return 0


for file in files:

    filepath = (directory + file)
    cap = cv2.VideoCapture(filepath)
    start_ret, start_frame = cap.read()


    processing(cap)

    cap.release()
    cv2.destroyAllWindows()