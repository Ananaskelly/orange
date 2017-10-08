import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import main
import assistance
import skimage.filters as filters


def hls_select(img, sthresh=(0, 255),lthresh=()):

    # 1) Convert to HLS color space
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    print(hls_img)
    # 2) Apply a threshold to the S channel
    L = hls_img[:,:,1]
    S = hls_img[:,:,2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(S)
    binary_output[(S >= sthresh[0]) & (S <= sthresh[1])
                 & (L > lthresh[0]) & (L <= lthresh[1])] = 1
    return binary_output

path = 'D:\\visionhack\\trainset\\trainset'
filename = '\\akn.204.006.left.avi'
filename2 = '\\akn.204.006.txt'

f = open(path + filename2, 'r')
line = f.readline().split(' ')
x = int(line[0])
y = int(line[1])

cap = cv2.VideoCapture(path + filename)
# hog = cv2.HOGDescriptor()

for i in range(110):
    ret, frame = cap.read()
frame_orig = frame
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert from BGR to LAB color space
#l, a, b = cv2.split(lab)  # split on 3 different channels
#l2 = clahe.apply(l)  # apply CLAHE to the L-channel
#lab = cv2.merge((l2,a,b))  # merge channels
#frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
frame = assistance.persp_transformation(frame, y)[0]
clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))

out = clahe.apply(frame)
cv2.imshow("out", out)

# mask = cv2.inRange(frame, (30, 0, 0), (50, 250, 250))

# res = cv2.bitwise_and(frame, frame, mask=mask)

# cv2.imshow("in", frame)
# cv2.imshow("out", res)
# cv2.rectangle(frame, (x, y), (x, y), 10)
# frame = cv2.resize(frame, (800, 600))


"""gt = assistance.persp_transformation(frame, y)
height = gt[0].shape[0]
width = gt[0].shape[1]

area_bound = width*height*0.005
height = gt[0].shape[0]
bottom_fix = int(height*0.2)
fixed = gt[0]
fixed = fixed[:height - bottom_fix, :]
mean = np.mean(fixed)
print(mean)
cv2.imshow("inp", gt[0])
edges = cv2.Canny(gt[0], 80, 250)
cv2.imshow("ololo", edges)
cv2.imwrite("test.jpg", edges)
ret2,th2 = cv2.threshold(fixed,mean,255,cv2.THRESH_BINARY)
cv2.imshow("ololo_bin", th2)
cv2.imwrite("test2.jpg", th2)
im2, contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour) < area_bound:
        continue
    epsilon = 0.1 * cv2.arcLength(contour, True)
    # approx = cv2.approxPolyDP(contour, epsilon, True)
    hull = cv2.convexHull(contour)
    cv2.drawContours(gt[0], [hull], -1, (0, 0, 255), 1)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #x, y, w, h = cv2.boundingRect(contour)
    # cv2.rectangle(gt[0], (x, y), (x + w, y + h), (0, 255, 0), 2)
    angle = math.degrees(math.atan2(box[0][0] - box[1][0], box[0][1] - box[1][1]))
    cv2.line(gt[0], (box[0][0], box[0][1]), (box[2][0], box[2][1]), (0, 255, 255), 2)
    # print(box[0])
    # print(box, angle)
    # if angle < 5:
    cv2.drawContours(gt[0], [box], 0, (0, 0, 255), 2)
cv2.imshow("finally",gt[0])
cv2.imwrite("finally.jpg", gt[0])
height = frame.shape[0]
height_roi = int(height*0.6)
bottom_fix = int(height*0.15)

t = assistance.persp_transformation(frame, y)
cv2.imshow("ololo", gt[0])
frame = frame[height_roi:height-bottom_fix,:]
th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,9,2)
th2 = cv2.bitwise_not(th2)
# gray_img = main.persp_transformation(frame)
# th2 = cv2.dilate(th2, np.ones((3,3), dtype='uint8'), iterations=1)
edges = cv2.Canny(frame, 130, 250)
# edges = cv2.dilate(edges, np.ones((3, 3), dtype='uint8'), iterations=1)


# edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype='uint8'))
cv2.imshow("a", edges)
lines = cv2.HoughLinesP(th2, 1, np.pi/360, threshold=100, minLineLength=20, maxLineGap=0)
# print(lines.shape)"""
"""for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),1)"""
"""im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
for contour in contours:
    epsilon = 0.1 * cv2.arcLength(contour, True)
    # approx = cv2.approxPolyDP(contour, epsilon, True)
    hull = cv2.convexHull(contour)
    cv2.drawContours(frame, [hull], -1, (0, 0, 255), 1)
    # rect = cv2.minAreaRect(hull)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    x, y, w, h = cv2.boundingRect(contour)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # angle = math.degrees(math.atan2(box[0][0] - box[1][0], box[0][1] - box[1][1]))
    # print(angle)
    # if angle < 5:
    # cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
cv2.imshow('result', frame)"""
cv2.waitKey()


