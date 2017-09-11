import numpy as np
import cv2
import time
import os, re

start_time = time.time()
directory = 'C:/Users/janch/Desktop/trainset/'
r = re.compile(".*avi")
files = filter(r.match, [f for f in os.listdir(directory)])


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


def processing(capture):
    frame_skip = 1
    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        try:
            if frame_count % frame_skip == 0:
                    ret, frame = cap.read()
            cv2.imshow('transformed', rs(persp_transformation(frame)[0]))
            cv2.imshow('frame', rs(frame))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(-1)
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