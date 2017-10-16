import cv2
import re
import os
import time

import detect_sign as sign_detector
import bridge_tunnel_wipes_detection as jan_detector


# directory = 'D:\\visionhack\\trainset\\'
directory = 'D:\\visionhack\\testset\\'
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
    jan_detector.old_gray_frame = 0
    jan_detector.p0 = 0
    jan_detector.searchin_charcts = []
    firsts = []
    seconds = []
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break

        current_tracked, current_values = sign_detector.blue_mask(frame, current_tracked, current_values)

        if jan_detector.opt_flow2(frame, frame_count) == 1:
            screen_wipers = '1'

        if bridge == '0':
            if jan_detector.bridge_tunnel(frame, firsts, seconds, frame_count) == 1:
                bridge = '1'

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(-1)
            break

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
