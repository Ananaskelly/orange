import cv2
import numpy as np


def load_and_process(path, x1, y1, x2, y2, is_grayscale=False):
    img = cv2.imread(path, 1)
    img_roi = img[x1:x2, y1:y2, :]
    img_roi = cv2.resize(img_roi, (32, 32))

    # test purpose
    # cv2.imshow('img', img_roi)
    # cv2.waitKey()

    if is_grayscale:
        img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

    img_out = np.multiply(img_roi, 1.0 / 255.0)
    return img_out


def load_and_process_without_roi(path, is_grayscale=False):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (32, 32))

    # test purpose
    # cv2.imshow('img', img_roi)
    # cv2.waitKey()

    if is_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_out = np.multiply(img, 1.0 / 255.0)
    return img_out


def load_and_process_image(img, is_grayscale=False):
    img = cv2.resize(img, (32, 32))

    # test purpose
    # cv2.imshow('img', img_roi)
    # cv2.waitKey()

    if is_grayscale:
        img_roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_out = np.multiply(img, 1.0 / 255.0)
    return img_out
