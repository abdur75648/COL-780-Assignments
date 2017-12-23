import numpy as np
import cv2
from functions import *

def get_gradient(img_gray):
    dx = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3)
    dy = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=3)
    grad = np.empty((img_gray.shape[0],img_gray.shape[1],1,2))
    grad[:,:,0,0] = dx
    grad[:,:,0,1] = dy
    return grad

def get_jacobian(x,y):
    j = np.zeros((x,y,6,2))
    j[:,:,0,0] = np.arange(y).reshape(1,-1).repeat(x,axis=0)
    j[:,:,1,1] = np.arange(y).reshape(1,-1).repeat(x,axis=0)
    j[:,:,2,0] = np.arange(x).reshape(-1,1).repeat(y,axis=1)
    j[:,:,3,1] = np.arange(x).reshape(-1,1).repeat(y,axis=1)
    j[:,:,4,0] = np.ones((x,y))
    j[:,:,5,1] = np.ones((x,y))
    return j

def get_hessian(steep_d,steep_d_T):
    m = np.einsum('ijkl,ijlm->ijkm',steep_d_T,steep_d)
    h = np.sum(np.sum(m,axis=0),axis=0)
    h_inv = np.linalg.pinv(h)
    return h, h_inv

def lk(img,template,max_iter=500,min_norm=0.01,verbose=True):
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(np.float32)
    template_gray = cv2.cvtColor(template,cv2.COLOR_RGB2GRAY).astype(np.float32)
    dt = get_gradient(template_gray)
    j = get_jacobian(img_gray.shape[0],img_gray.shape[1])
    steep_d = np.einsum('ijkl,ijml->ijkm',dt,j)
    steep_d_T = np.rollaxis(steep_d,3,2)
    h, h_inv = get_hessian(steep_d,steep_d_T)
    param = np.array([[0,0,0],[0,0,0]],dtype=np.float32)
    param_1 = np.array([[1,0,0],[0,1,0]],dtype=np.float32)
    param_norm = 1
    iter = 0
    while param_norm > min_norm and iter < max_iter:
        img_warp = cv2.warpAffine(img_gray,param+param_1,(img_gray.shape[1],img_gray.shape[0])).astype(np.float32)
        # show_img("Warp",img_warp.astype(np.uint8),0)
        error_img = (img_warp - template_gray).reshape((img_gray.shape[0],img_gray.shape[1],1,1))
        summ_m = np.einsum('ijkl,ijlm->ijkm',steep_d_T,error_img)
        summ = np.sum(np.sum(summ_m,axis=0),axis=0)
        dp = np.dot(h_inv,summ)
        param_copy = param.copy()
        param[0,0] = param_copy[0,0] + dp[0] + param_copy[0,0]*dp[0] + param_copy[0,1]*dp[1]
        param[1,0] = param_copy[1,0] + dp[1] + param_copy[1,0]*dp[0] + param_copy[1,1]*dp[1]
        param[0,1] = param_copy[0,1] + dp[2] + param_copy[0,0]*dp[2] + param_copy[0,1]*dp[3]
        param[1,1] = param_copy[1,1] + dp[3] + param_copy[1,0]*dp[2] + param_copy[1,1]*dp[3]
        param[0,2] = param_copy[0,2] + dp[4] + param_copy[0,0]*dp[4] + param_copy[0,1]*dp[5]
        param[1,2] = param_copy[1,2] + dp[5] + param_copy[1,0]*dp[4] + param_copy[1,1]*dp[5]
        param_norm = np.linalg.norm(dp)
        iter = iter + 1
        if verbose: print(iter," - ",param_norm)
    print(iter)
    img_warp = cv2.warpAffine(img_gray,param+param_1,(img_gray.shape[1],img_gray.shape[0]))
    return img_warp.astype(np.uint8)

if __name__ == "__main__":
    img = load_img("img2.jpg",1)
    template = load_img("template2.jpg",1)
    warp = lk(img,template)
    show_multiple_img(["Image","Template","Warp"],[img,template,warp],0)
    save_img("warp2",warp)