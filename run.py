import cv2
import re
import os
import time

import detect_sign as sign_detector
import bridge_tunnel_wipes_detection as jan_detector


directory = 'D:\\visionhack\\trainset\\trainset\\'
r = re.compile(".*avi")


def processing(cap):
    frame_skip = 1
    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
    start_ret, start_frame = cap.read()

    bridge = '0'
    city = '0'
    city_2 = '0'
    road_bump = '0'
    screen_wipers = '0'
    zebra = '0'
    current_tracked = []
    current_values = []
    old_gray_frame = 0
    p0 = 0
    searchin_charcts = []
    firsts = []
    seconds = []
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break

        current_tracked, current_values = sign_detector.blue_mask(frame, current_tracked, current_values)

        if jan_detector.opt_flow2(frame, old_gray_frame, p0, searchin_charcts) == 1:
            screen_wipers = '1'
        if frame_count % 70 == 0:
            old_gray_frame = 0
            p0 = 0
        if bridge == 0:
            if jan_detector.bridge_tunnel(frame, firsts, seconds, frame_count) == 1:
                bridge = '1'

        frame_count += 1

    if 80 in current_values:
        road_bump = '1'

    if 79 in current_values:
        zebra = '1'

    with open('answer.txt', 'a') as f:
        f.write(file + ' ' + bridge + city + city_2 + road_bump + screen_wipers + zebra + '\n')
    cap.release()
    cv2.destroyAllWindows()


start_time = time.time()
files = filter(r.match, [f for f in os.listdir(directory)])
for file in files:
    file_path = (directory + file)
    capture = cv2.VideoCapture(file_path)
    processing(capture)

    print("Time: " + str(time.time() - start_time))
