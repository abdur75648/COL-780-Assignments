import numpy as np
import cv2
from functions import *
from lk_track import *

if __name__ == "__main__":
    vid = cv2.VideoCapture("Test Data/vid10.avi")
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print(length," - ",fps)
    ret, frame = vid.read()
    template = frame
    template_gray = cv2.cvtColor(template,cv2.COLOR_RGB2GRAY)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("out/stable_vid10.avi",fourcc,fps,(template_gray.shape[1],template_gray.shape[0]))
    out.write(np.repeat(template_gray[...,None],3,axis=2))
    frame_count = 1
    while(1):
        ret, frame = vid.read()
        print(frame_count,end=' - ')
        if not ret:
            break
        frame_warp = lk(frame,template,500,0.001,False)
        # show_img("Warp",frame_warp,0)
        frame_count = frame_count + 1
        out.write(np.repeat(frame_warp[...,None],3,axis=2))