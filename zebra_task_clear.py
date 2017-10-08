import cv2
import numpy as np
import assistance as assist
import math


DELTA = 25
DELTA_AMOUNT = 4


def process_cap(cap, txt_file):
    frame_skip = 1
    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
    start_ret, start_frame = cap.read()
    f = open(txt_file, 'r')
    line = f.readline().split(' ')
    y = int(line[1])
    while cap.isOpened():
        if frame_count % frame_skip == 0:
            ret, frame = cap.read()
            if frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            transformed = assist.persp_transformation(frame, y)[0]

            m = np.mean(transformed)
            m += int(m * 0.4)

            # ret2, binarized = cv2.threshold(transformed, m, 255, cv2.THRESH_BINARY)
            height = transformed.shape[0]
            bottom_fix = int(height * 0.2)

            roi = transformed[:height - bottom_fix, :]
            answ, img, ang = analyze(roi, m)
            if answ:
                print(ang)
                cv2.imwrite("aa.jpg", roi)
                return '1'
                # cv2.imshow('orig', rs(frame))
                # cv2.imshow('transformed', rs(persp_transformation(frame)[0]))
                # cv2.imshow('frame', rs(frame))
                # if cv2.waitKey(30) & 0xFF == ord('q'):
                #    print(-1)
                #    break
        return '0'


def get_and_analyze_vertical_projection(img):
    height = img.shape[0]
    projection = np.count_nonzero(img, axis=0)
    max_vals = []
    bins_length = []
    abs_max = 0
    current_max = 0
    current_width = 0
    for el in projection:
        if el == 0:
            if current_max != 0:
                abs_max = max(abs_max, current_max)
                max_vals.append(current_max)
                bins_length.append(current_width)
                current_max = 0
                current_width = 0
        else:
            current_max = max(current_max, el)
            current_width += 1

    count_near_max = 0
    for val in max_vals:
        if val > height*0.2:
            count_near_max += 1
    return count_near_max >= DELTA_AMOUNT


def analyze(roi, mean):
    ret2, th2 = cv2.threshold(roi, 170, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    th2 = cv2.dilate(th2, kernel, iterations=1)
    height = roi.shape[0]
    width = roi.shape[1]
    area_bound_2 = width*height*0.1
    area_bound = width*height*0.005
    im2, contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angle_mean = []
    angle_2_mean = []
    amount = []
    length = []
    for contour in contours:
        if cv2.contourArea(contour) < area_bound or cv2.contourArea(contour) > area_bound_2:
            continue
        epsilon = 0.1 * cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(contour)
        cv2.drawContours(th2, [hull], -1, (255, 255, 255), 1)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # x, y, w, h = cv2.boundingRect(hull)
        # cv2.rectangle(th2, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.drawContours(th2, [box],0, (255,255,255), 2)
        # cv2.line(th2, (box[0][0], box[0][1]), (box[2][0], box[2][1]), (255, 255, 255), 2)
        angle = math.degrees(math.atan2(box[0][0] - box[2][0], box[0][1] - box[2][1]))
        curr_length = math.sqrt((box[0][0] - box[2][0])**2 + (box[0][1] - box[2][1])**2)

        box1 = [box[np.argsort(box[:,0])]][0]
        angle_2 = math.degrees(math.atan2(box[2][0] - box[3][0], box[2][1] - box[3][1]))

        if len(angle_mean) == 0:
            angle_mean.append(angle)
            angle_2_mean.append(angle_2)
            amount.append(1)
            length.append(curr_length)
        else:
            flag = False
            for (ind, curr_angle) in enumerate(angle_mean):
                if curr_angle - 15 < angle < curr_angle + 15:
                    amount[ind] += 1
                    angle_mean[ind] = (angle_mean[ind] + angle) / amount[ind]
                    angle_2_mean[ind] = (angle_2_mean[ind] + angle_2) / amount[ind]
                    length[ind] = max(curr_length, length[ind])
                    flag = True
            if not flag:
                angle_mean.append(angle)
                angle_2_mean.append(angle_2)
                amount.append(1)
                length.append(curr_length)
    if len(amount) == 0:
        return False, th2, 0
    return max(amount) > DELTA_AMOUNT, th2, angle_2_mean[amount.index(max(amount))]
