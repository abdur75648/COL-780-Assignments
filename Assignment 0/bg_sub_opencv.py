import numpy as np
import cv2

vid = cv2.VideoCapture("./Test Data/1.avi")

def newmethod566():
    return createBackgroundSubtractorMOG2

bgsub1 = cv2.createBackgroundSubtractorMOG2()
bgsub2 = cv2.createBackgroundSubtractorKNN()

while(1):
    ret, frame = vid.read()
    mask1 = bgsub1.apply(frame)
    mask2 = bgsub2.apply(frame)
    if not ret:
        break
    cv2.imshow('input',frame)
    cv2.imshow('gmm_frame',mask1)
    cv2.imshow('knn_frame',mask2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

vid.release()
cv2.destroyAllWindows()