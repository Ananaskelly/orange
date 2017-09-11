import cv2
import numpy as np


path = 'D:\\visionhack\\trainset\\trainset'
filename = '\\akn.395.047.left.avi'

cap = cv2.VideoCapture(path + filename)
# hog = cv2.HOGDescriptor()

for i in range(180):
    ret, frame = cap.read()
width = frame.shape[1]
roi_width = int(width*0.5)
frame = cv2.resize(frame, (800, 600))
# frame = frame[:, width - roi_width:, :]

# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# res, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow('ololo', frame)
"""
print(width - roi_width)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
sensitivity = 150
lower_white = np.array([0,0,0])
upper_white = np.array([0,0,255])

# Threshold the HSV image to get only white colors
mask = cv2.inRange(hsv, lower_white, upper_white)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame, frame, mask=mask)

cv2.imshow('frame', frame)
cv2.imshow('mask', mask)
cv2.imshow('res', res)"""



"""frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

cv2.imshow('first', frame)"""

"""clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(5, 5))

lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
l, a, b = cv2.split(lab)  # split on 3 different channels

l2 = clahe.apply(l)  # apply CLAHE to the L-channel

lab = cv2.merge((l2,a,b))  # merge channels
img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
cv2.imshow('Increased contrast', img2)

frame = img2
# cv2.imshow("inp", frame)
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# frame = cv2.GaussianBlur(frame, (3, 3), 0)

cv2.imshow("lol", frame)"""
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,5,10)
cv2.imshow('th2', th2)
edges = cv2.Canny(frame, 150, 250)
edges = cv2.dilate(edges, np.ones((3,3), dtype='uint8'),iterations = 1)
cv2.imshow("lol", edges)
edges =cv2.bitwise_not(th2)
# closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
# cv2.imshow("closed", closing)
"""im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    epsilon = 0.2 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, False)
    cv2.drawContours(frame, [approx], -1, (0, 0, 255), 1)

cv2.imshow('out', frame)"""


# frame = cv2.GaussianBlur(frame, (3, 3), 11)

# edges = cv2.Canny(frame, 50, 150, apertureSize=3)

# ret2, th2 = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)
# cv2.imshow('otsu', th2)


# cv2.imshow('edges', edges)
#edges = cv2.bitwise_not(edges)
#cv2.imshow("a", edges)

lines = cv2.HoughLinesP(edges, 1, np.pi/360, threshold=50, minLineLength=10, maxLineGap=5)
print(lines.shape)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),1)

cv2.imshow('result', frame)
# sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=15)
# sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=15)

# answ = cv2.bitwise_and(sobelx, sobely)

# cv2.imshow('frame', answ)


cap.release()
cv2.waitKey()
cv2.destroyAllWindows()

