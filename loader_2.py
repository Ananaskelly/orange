import cv2
import re
import os
import time
import detect_sign as detector
import prepare_set
import time
import numpy as np
import gc
start_time = time.time()

# directory = 'C:\\trainset\\'
# directory = 'D:\\testset\\'
directory = 'D:/visionhack/trainset/trainset/'
info_dir = 'D:/visionhack/trainset/trainset/train.txt'


r = re.compile(".*avi")
r2 = re.compile(".*txt")


def processing(cap, foe_value=0, flag='0'):
    start_ret, start_frame = cap.read()

    current_tracked = []
    current_values = []
    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
    firsts = []
    seconds = []
    bridge = '0'
    min_v = 1000
    max_v = 0
    print(flag)

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        if frame_count%10 != 0:
            frame_count += 1
            continue
        # current_tracked, current_values = detector.blue_mask(frame, current_tracked, current_values)
        # line_detector.process_frame(frame, foe_value, min_v, max_v)
        prepare_set.prepare(frame, foe_value, flag)
        # if bridge == '0':
        #    if histogram(frame, firsts, seconds, frame_count) == 1:
        #        bridge = '1'
        frame_count += 1
    if 80 in current_values:
            bump = '1'
    else:
        bump = '0'
    if 79 in current_values:
        zebra = '1'
    else:
        zebra = '0'
    print(file + ' ' + bridge + '00' + bump + '0' + zebra + '\n')
    print(min_v, max_v)
    with open('answer.txt', 'a') as f:
        f.write(file + ' ' + bridge + '00' + bump + '0' + zebra + '\n')
    cap.release()
    cv2.destroyAllWindows()


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
        if abs(sum(firsts) - sum(seconds)) > 100:
            print('trololololo')
            return 1

    return 0


f = open(info_dir)
files = filter(r.match, [f for f in os.listdir(directory)])
foe_files = list(filter(r2.match, [f for f in os.listdir(directory)]))
for ind, file in enumerate(list(files)):
    curr_info = (f.readline()).rstrip().split(' ')
    print(curr_info[0])
    flag = list(curr_info[1])[-1]

    file_path = (directory + file)

    foe_txt = open(directory + foe_files[ind], 'r')
    foe_value = int(foe_txt.readline().split(' ')[1])

    capture = cv2.VideoCapture(file_path)
    processing(capture, foe_value, flag)
    capture.release()
    cv2.destroyAllWindows()

