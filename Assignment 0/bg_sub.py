import numpy as np
import cv2

def get_gaussian(diff_sq_sum,mu,sigma_sq):
    p = (2*np.pi)**(mu.shape[2]/2)
    exponent = -0.5*diff_sq_sum/sigma_sq
    gaussians = np.exp(exponent)/(p*(sigma_sq**0.5))
    return gaussians

def get_background_gaussians(w,sigma,T):
    w_ratio = -1*w/sigma
    sorted_ratio_idx = np.argsort(w_ratio,axis=2)
    w_ratio.sort(axis=2)
    ratio_cumsum = np.cumsum(-1*w_ratio,axis=2)
    threshold_mask = (ratio_cumsum < T)
    background_gaussian_mask = np.choose(np.rollaxis(sorted_ratio_idx,axis=2),np.rollaxis(threshold_mask,axis=2))
    return np.rollaxis(background_gaussian_mask,axis=0,start=3)

def get_masks(background_gaussians_mask,diff_sq_sum,lambda_sq,sigma_sq):
    update_mask = background_gaussians_mask*(diff_sq_sum/sigma_sq < lambda_sq*sigma_sq)
    foreground_mask = ~np.any(update_mask,axis=2)
    mask_img = np.array(foreground_mask*255,dtype=np.uint8)
    replace_mask = np.repeat(foreground_mask[...,None],background_gaussians_mask.shape[2],axis=2)
    return update_mask, replace_mask, mask_img

def update(gaussians,alpha,w,mu,sigma_sq,diff_sq_sum,update_mask,replace_mask):
    replace_mask_extended = np.repeat(replace_mask[:,:,None,:],mu.shape[2],axis=2)
    update_mask_extended = np.repeat(update_mask[:,:,None,:],mu.shape[2],axis=2)
    w = (1-alpha)*w + alpha*update_mask
    w[replace_mask] = 0.0001
    rho = alpha*gaussians
    rho_extended = np.repeat(rho[:,:,None,:],mu.shape[2],axis=2)
    mu[update_mask_extended] = (1-rho_extended[update_mask_extended])*mu[update_mask_extended] + rho_extended[update_mask_extended]*np.repeat(frame[...,None],mu.shape[3],axis=3)[update_mask_extended]
    mu[replace_mask_extended] = np.repeat(frame[...,None],mu.shape[3],axis=3)[replace_mask_extended]
    sigma_sq[replace_mask] = 16
    sigma_sq[update_mask] = (1-rho[update_mask])*sigma_sq[update_mask] + rho[update_mask]*diff_sq_sum[update_mask]
    sigma = np.sqrt(sigma_sq)
    return w, mu, sigma_sq, sigma
    
# Testing
if __name__ == '__main__':
    vid = cv2.VideoCapture("./Test Data/2.avi")
    # vid = cv2.VideoCapture(0)
    ret, frame = vid.read()
    bgsub1 = cv2.createBackgroundSubtractorMOG2()
    bgsub2 = cv2.createBackgroundSubtractorKNN()

    # Initial values of parameters
    K = 3
    lambda_sq = 2.5**2
    alpha = 0.2
    T = 0.7
    w = np.full((frame.shape[0],frame.shape[1],K),1/K)
    mu = np.zeros(frame.shape+tuple([K]))
    sigma = np.ones(w.shape)
    sigma_sq = sigma
    diff = frame[...,None] - mu
    diff_sq_sum = np.sum(diff*diff,axis=2)

    while(1):
        if not ret:
            break
        gaussians = get_gaussian(diff_sq_sum,mu,sigma_sq)
        background_gaussians_mask = get_background_gaussians(w,sigma,T)
        update_mask, replace_mask, mask_img = get_masks(background_gaussians_mask,diff_sq_sum,lambda_sq,sigma_sq)
        mask1 = bgsub1.apply(frame)
        mask2 = bgsub2.apply(frame)
        cv2.imshow('Input',frame)
        cv2.imshow('Mask',mask_img)
        cv2.imshow('GMM_OpenCV',mask1)
        cv2.imshow('KNN_OpenCV',mask2)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        ret, frame = vid.read()
        diff = frame[...,None] - mu
        diff_sq_sum = np.sum(diff*diff,axis=2)
        w, mu, sigma_sq, sigma = update(gaussians,alpha,w,mu,sigma_sq,diff_sq_sum,update_mask,replace_mask)      

    vid.release()
    cv2.destroyAllWindows()