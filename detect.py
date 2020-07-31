import numpy as np
import cv2
import argparse
import os
from scipy import ndimage
from keras.models import load_model
from random import choice
import pyautogui
import preprocessing


img_rows, img_cols = 224, 224

cap = cv2.VideoCapture(0)
model = load_model("models/model5.hdf5")
font = cv2.FONT_HERSHEY_SIMPLEX
c=0
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.rectangle(frame, (0, 100), (300, 400), (255, 255, 255), 3)
    cv2.imwrite("images/frame0.jpg", frame[100:400, 0:300])
    path = "images/frame0.jpg"
    orig = cv2.imread(path)
    image = preprocessing.pre_process_image(path)
    orig, image = preprocessing.find_corners_of_largest_polygon(image, orig)
    img = image
    image = np.array(image).reshape(-1, img_rows, img_cols, 1)
    pred = np.argmax(model.predict(image))
    num_zeros = np.count_nonzero(img)
    print("num: ", num_zeros)
    if num_zeros > 20000 :
        pred = 2
    print(pred, " ", c)
    #c = c+1
    if pred == 1:
        pyautogui.press('space')

    cv2.imshow("frame", frame)
    cv2.imshow("detect", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()












