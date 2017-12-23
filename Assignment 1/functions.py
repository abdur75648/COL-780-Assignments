import numpy as np
import cv2
from scipy.stats import entropy as scipy_entropy

def load_img(name: str,mode: int) -> np.ndarray:
    return cv2.imread("Test Data/"+name, mode)

def show_img(title: str,img: np.ndarray,wait: int) -> int:
    cv2.imshow(title, img)
    k = cv2.waitKey(wait)
    cv2.destroyWindow(title)
    return k

def show_multiple_img(title: list,img: list,wait: int) -> int:
    if type(title) != list or len(title) == 0:
        title = [str(x) for x in range(len(img))]
    for x, y in zip(title, img):
        cv2.imshow(x,y)
    k = cv2.waitKey(wait)
    for x in title:
        cv2.destroyWindow(x)
    return k

def save_img(name,img):
    if type(name) == str:
        cv2.imwrite("out/"+name+".jpg",img)
    else:
        [cv2.imwrite("out/"+n+".jpg",i) for n,i in zip(name,img)]