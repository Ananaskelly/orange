import cv2
import numpy as np
import assistance


def process_frame(frame, foe_value, min_v, max_v):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    road = assistance.persp_transformation(gray_frame, foe_value)[0]
    road_res = cv2.resize(road, (800, 640))
    road_res = cv2.boxFilter(road_res, -1, (7, 7))
    road_bin = cv2.adaptiveThreshold(road_res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 2)

    road_bin = cv2.bitwise_not(road_bin)
    road_bin = cv2.dilate(road_bin, np.ones((7, 7)), iterations=1)

    cv2.imshow("transformed", road_bin)
    cv2.waitKey(1)
