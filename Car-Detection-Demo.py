import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

vid = cv.VideoCapture('video/Vehicles ongoing traffic.mp4')
if (vid.isOpened() == False):
    print("ERROR opening the video file")
hight = 1080
width = 720
car_cascade = cv.CascadeClassifier('cars.xml')

while (vid.isOpened()):
    ret, frame = vid.read()
    if ret == True:
        # '1' property ID of CV_CAP_PROP_POS_FRAME (takes current frame number).
        if vid.get(1) % 5 == 0:
            # convert video to grayscale.
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            cars = car_cascade.detectMultiScale(frame, 1.1, 1)
            # Draw border around the car
            for (x, y, w, h) in cars:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 256), 2)
            cv.imshow('Demo Test', frame)
            key = cv.waitKey(30)
            if key == ord('q'):
                break
    else:
        break

vid.release()
cv.destroyAllWindows()