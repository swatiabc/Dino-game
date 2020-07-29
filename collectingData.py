import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("label")
parser.add_argument("count")
args = parser.parse_args()

label = os.path.join("test", args.label)
count = args.count

print(label, " ", count)

cap = cv2.VideoCapture(0)
#cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
c = 0
flag = False
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    if c == int(count):
        break
    cv2.rectangle(frame, (0, 100), (300, 400), (255, 255, 255), 3)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if flag:
        path = os.path.join(label, '{}.jpg'.format(c))
        cv2.imwrite(path, frame[100:400, 0:300])
        c = c+1
    if key == ord('a'):
        flag = not flag
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
