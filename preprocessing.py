import numpy as np
import cv2
import os
from scipy import ndimage
import operator


img_rows, img_cols = 224, 224


def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def shift_according_to_center_of_mass(img):
    img = cv2.bitwise_not(img)
    shiftx, shifty = get_best_shift(img)
    shifted = shift(img, shiftx, shifty)
    img = cv2.bitwise_not(img)
    return img


def pre_process_image(img):
    image_array = cv2.imread(os.path.join(img), cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(image_array, (img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)
    _, resized_array = cv2.threshold(resized_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized_array = shift_according_to_center_of_mass(resized_array)
    return resized_array


def find_corners_of_largest_polygon(img, orig):
    contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]
    img = np.zeros(img.shape)
    #orig = cv2.drawContours(orig, [polygon], 0, (0, 255, 0), 3)
    #img = cv2.drawContours(img, [polygon], 0, (225, 255, 225), 3)
    img = cv2.fillPoly(img, [polygon], (255, 255, 255))
    return orig, img


# path = "dataset/fist/22.jpg"
# orig_image = cv2.imread(path)
# image = pre_process_image(path)
# orig, image = find_corners_of_largest_polygon(image, orig_image)
# cv2.imshow("image", image)
# cv2.imshow("orig",orig)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

