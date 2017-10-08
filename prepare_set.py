import cv2
import assistance
import os


bound = 0.2

true_dir = 'D:/visionhack/trainset/true/'
false_dir = 'D:/visionhack/trainset/false/'
bias = 400000


def prepare(frame, foe_value, flag):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    road = assistance.persp_transformation(gray_frame, foe_value)[0]
    width_bound = int(road.shape[1]*bound)
    height_bound = int(road.shape[0]*bound*1.5)
    road = road[height_bound:-height_bound, width_bound:-width_bound]
    road = cv2.resize(road, (224, 224))

    if flag == '0':
        list_files = os.listdir(false_dir)
        ind = len(list_files)
        road = cv2.resize(road, (224, 224))
        cv2.imwrite(false_dir + str(ind) + '.jpg', road)
    else:
        list_files = os.listdir(true_dir)
        ind = len(list_files)
        road = cv2.resize(road, (224, 224))
        cv2.imwrite(true_dir + str(ind + bias) + '.jpg', road)
    # cv2.imshow("transformed", road)
    # cv2.waitKey(1)
