import cv2
import numpy as np
import math
import copy
import recognition.recognition as rec


# delta bound between contours in frames
delta_bound = 20
border = 0
lower = np.array([86, 31, 4], dtype='uint8')
upper = np.array([220, 88, 50], dtype='uint8')
kernel = np.ones((19, 19), np.uint8)


# get angle between crossing lines
def get_angle(arr):
    dx12 = arr[0][0][0] - arr[1][0][0]
    dx31 = arr[2][0][0] - arr[1][0][0]
    dy21 = arr[0][0][1] - arr[1][0][1]
    dy31 = arr[2][0][1] - arr[1][0][1]
    m12 = math.sqrt(dx12*dx12 + dy21*dy21)
    m13 = math.sqrt(dx31*dx31 + dy31*dy31)
    angle = math.acos((dx12*dx31 + dy21*dy31)/(m12*m13))
    return math.degrees(angle)


class FuckingContour:
    data = np.empty([0, 0])
    data_bounds = []
    storage = []

    def __init__(self, array, bounds):
        self.data = array
        self.data_bounds = bounds
        self.storage = []
        self.delay = 0


def blue_mask(frame, current_tracked, current_values):

    mask = cv2.inRange(frame, lower, upper)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    dilated_mask = cv2.medianBlur(dilated_mask, 5)

    # cv2.imshow("masked", cv2.resize(dilated_mask, (800, 640)))

    im2, contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = copy.copy(current_tracked)
    current_tracked = []
    if len(candidates) == 0:
        for contour in contours:
            if cv2.contourArea(contour) < 45 * 45:
                continue
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                angle_1 = get_angle(approx[:3])
                angle_2 = get_angle(approx[1:])

                simplify_approx = [(row[0]).tolist() for row in approx]
                X = [row[0] for row in simplify_approx]
                Y = [row[1] for row in simplify_approx]
                w = max(X) - min(X)
                h = max(Y) - min(Y)
                if abs(angle_1 - 90) < 10 and abs(angle_2 - 90) < 10 and 0.8 < h / w < 1.2:
                    fc = FuckingContour(frame[min(Y):min(Y)+h, min(X):min(X)+w], [min(X), min(Y), w, h])
                    cv2.drawContours(frame, [approx], 0, (0, 255, 255), 10)
                    current_tracked.append(fc)
    else:
        for contour in contours:
            if cv2.contourArea(contour) < 45 * 45:
                continue
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                angle_1 = get_angle(approx[:3])
                angle_2 = get_angle(approx[1:])

                simplify_approx = [(row[0]).tolist() for row in approx]
                X = [row[0] for row in simplify_approx]
                Y = [row[1] for row in simplify_approx]
                w = max(X) - min(X)
                h = max(Y) - min(Y)
                if abs(angle_1 - 90) < 10 and abs(angle_2 - 90) < 10 and 0.8 < h / w < 1.2:
                    # obj = frame[min(Y):max(Y), min(X):max(X)]
                    # num = rec.get_class_no(obj)
                    flag = False
                    for candidate in candidates:
                        if abs(candidate.data_bounds[0] - min(X)) < delta_bound and abs(candidate.data_bounds[1] - min(Y)) < \
                                delta_bound and abs(candidate.data_bounds[2] - w) < delta_bound and abs(candidate.data_bounds[3] - h) \
                                < delta_bound:
                            candidates.remove(candidate)
                            new_image = candidate.data
                            candidate.delay = 0
                            candidate.storage.append(new_image)
                            curr_length = len(candidate.storage)
                            # little clean up
                            if curr_length > 5:
                                candidate.storage.pop(0)
                            candidate.data = frame[min(Y):min(Y)+h, min(X):min(X)+w]
                            candidate.data_bounds = [min(X), min(Y), w, h]
                            current_tracked.append(candidate)
                            cv2.drawContours(frame, [approx], 0, (255, 255, 255), 10)
                            flag = True
                            break
                    if not flag:
                        fc = FuckingContour(frame[min(Y):min(Y) + h, min(X):min(X) + w], [min(X), min(Y), w, h])
                        current_tracked.append(fc)
        for candidate in candidates:
            if candidate.delay < 2:
                candidate.delay += 1
                current_tracked.append(candidate)
                continue
            if len(candidate.storage) < 2:
                continue
            ind = min(len(candidate.storage) - 1, 5)

            rt = candidate.storage[len(candidate.storage) - ind]
            h = rt.shape[0]
            w = rt.shape[1]
            # print(rt.shape)
            if h < 45 or w < 45 or np.std(rt) < 30:
                continue
            num, p = rec.get_class_no(rt)

            if p < 0.98:
                continue

            cv2.imshow(str(num), rt)
            cv2.waitKey()
            current_values.append(num)

    cv2.imshow("orig", cv2.resize(frame, (400, 320)))
    cv2.waitKey(1)
    return current_tracked, current_values
