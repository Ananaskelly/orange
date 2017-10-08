import cv2
import numpy as np


def persp_transformation(img, y_val):
    x = img.shape[1]
    y = img.shape[0]
    img_size = (x, y)
    bound = y*0.078
    source_points = np.float32([
        [0.117 * x, y],
        [(0.5 * x) - (x*0.23), y_val + bound],
        [(0.5 * x) + (x*0.23), y_val + bound],
        [x - (0.117 * x), y]]
    )

    destination_points = np.float32([
        [0, y],
        [0, 0],
        [x, 0],
        [x, y]
    ])

    persp_transform = cv2.getPerspectiveTransform(source_points, destination_points)
    inverse_persp_transform = cv2. getPerspectiveTransform(destination_points, source_points)

    warped_img = cv2.warpPerspective(img, persp_transform, img_size, flags=cv2.INTER_LINEAR)

    return warped_img, inverse_persp_transform