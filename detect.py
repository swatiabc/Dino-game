import numpy as np
import cv2
import argparse
import os
from scipy import ndimage
from keras.models import load_model
from random import choice
import pyautogui


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

    # Centralize the image according to center of mass
    shiftx, shifty = get_best_shift(img)
    shifted = shift(img, shiftx, shifty)
    img = shifted

    img = cv2.bitwise_not(img)
    return img


def pre_process_image(img):
    image_array = cv2.imread(os.path.join(img), cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(image_array, (img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)
    _, resized_array = cv2.threshold(resized_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized_array = shift_according_to_center_of_mass(resized_array)
    return resized_array


cap = cv2.VideoCapture(0)
#cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
model = load_model("models/model2.hdf5")
font = cv2.FONT_HERSHEY_SIMPLEX
c=0
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.rectangle(frame, (0, 100), (300, 400), (255, 255, 255), 3)
    cv2.imwrite("images/frame{}.jpg".format(0), frame[100:400, 0:300])
    image = pre_process_image("images/frame{}.jpg".format(0))
    image = np.array(image).reshape(-1, img_rows, img_cols, 1)
    pred = np.argmax(model.predict(image))
    #cv2.putText(frame, pred, (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
    print(pred, " ", c)
    c = c+1
    if pred == 1:
        pyautogui.press('space')
    elif pred == 2:
        pyautogui.press('down')

    cv2.imshow("frame", frame)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()












