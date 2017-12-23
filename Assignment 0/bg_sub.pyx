import numpy as np
import cv2
cimport numpy as cnp

cdef cnp.ndarray[cnp.float_t,ndim=3] get_gaussian(cnp.ndarray[cnp.float_t,ndim=3] diff_sq_sum,cnp.ndarray[cnp.float_t,ndim=4] mu,cnp.ndarray[cnp.float_t,ndim=3] sigma_sq):
    cdef float p = (2*np.pi)**(mu.shape[2]/2)
    cdef cnp.ndarray[cnp.float_t,ndim=3] exponent = -0.5*diff_sq_sum/sigma_sq
    cdef cnp.ndarray[cnp.float_t,ndim=3] gaussians = np.exp(exponent)/(p*(sigma_sq**0.5))
    return gaussians

cdef cnp.ndarray[cnp.float_t,ndim=3] get_background_gaussians(cnp.ndarray[cnp.float_t,ndim=3] w,cnp.ndarray[cnp.float_t,ndim=3] sigma,float T):
    cdef cnp.ndarray[cnp.float_t,ndim=3] w_ratio = -1*w/sigma
    cdef cnp.ndarray[cnp.int_t,ndim=3] sorted_ratio_idx = np.argsort(w_ratio,axis=2)
    w_ratio.sort(axis=2)
    cdef cnp.ndarray[cnp.float_t,ndim=3] ratio_cumsum = np.cumsum(-1*w_ratio,axis=2)
    cdef cnp.ndarray[cnp.uint8_t,ndim=3,cast=True] threshold_mask = (ratio_cumsum < T).astype(np.uint8)
    print(threshold_mask.dtype)
    cdef cnp.ndarray[cnp.uint8_t,ndim=3,cast=True] background_gaussian_mask = np.choose(np.rollaxis(sorted_ratio_idx,axis=2),np.rollaxis(threshold_mask,axis=2)).astype(np.uint8)
    return np.rollaxis(background_gaussian_mask,axis=0,start=3)

cdef tuple get_masks(cnp.ndarray[cnp.uint8_t,ndim=3,cast=True] background_gaussians_mask,cnp.ndarray[cnp.float_t,ndim=3] diff_sq_sum,float lambda_sq,cnp.ndarray[cnp.float_t,ndim=3] sigma_sq):
    print(background_gaussians_mask.dtype)
    cdef cnp.ndarray[cnp.uint8_t,ndim=3,cast=True] update_mask = background_gaussians_mask*(diff_sq_sum/sigma_sq < lambda_sq*sigma_sq)
    cdef cnp.ndarray[cnp.uint8_t,ndim=2,cast=True] foreground_mask = ~np.any(update_mask,axis=2)
    cdef cnp.ndarray[cnp.uint8_t,ndim=2] mask_img = np.array(foreground_mask*255,dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t,ndim=3,cast=True] replace_mask = np.repeat(foreground_mask[...,None],background_gaussians_mask.shape[2],axis=2)
    return update_mask, replace_mask, mask_img

cdef tuple update(cnp.ndarray[cnp.float_t,ndim=3] frame,cnp.ndarray[cnp.float_t,ndim=3] gaussians,float alpha,cnp.ndarray[cnp.float_t,ndim=3] w,cnp.ndarray[cnp.float_t,ndim=4] mu,cnp.ndarray[cnp.float_t,ndim=3] sigma_sq,cnp.ndarray[cnp.float_t,ndim=3] diff_sq_sum,cnp.ndarray[cnp.uint8_t,ndim=3,cast=True] update_mask,cnp.ndarray[cnp.uint8_t,ndim=3,cast=True] replace_mask):
    print(replace_mask.dtype)
    print(update_mask.dtype)
    cdef cnp.ndarray[cnp.uint8_t,ndim=4,cast=True] replace_mask_extended = np.repeat(replace_mask[:,:,None,:],mu.shape[2],axis=2)
    cdef cnp.ndarray[cnp.uint8_t,ndim=4,cast=True] update_mask_extended = np.repeat(update_mask[:,:,None,:],mu.shape[2],axis=2).astype(np.uint8)
    w = (1-alpha)*w + alpha*update_mask
    w[replace_mask] = 0.0001
    cdef cnp.ndarray[cnp.float_t,ndim=3] rho = alpha*gaussians
    cdef cnp.ndarray[cnp.float_t,ndim=4] rho_extended = np.repeat(rho[:,:,None,:],mu.shape[2],axis=2)
    cdef cnp.ndarray[cnp.float_t,ndim=4] frame_repeat = np.repeat(frame[...,None],mu.shape[3],axis=3)
    mu[update_mask_extended] = (1-rho_extended[update_mask_extended])*mu[update_mask_extended] + rho_extended[update_mask_extended]*frame_repeat[update_mask_extended]
    mu[replace_mask_extended] = frame_repeat[replace_mask_extended]
    sigma_sq[replace_mask] = 16
    sigma_sq[update_mask] = (1-rho[update_mask])*sigma_sq[update_mask] + rho[update_mask]*diff_sq_sum[update_mask]
    cdef cnp.ndarray[cnp.float_t,ndim=3] sigma = np.sqrt(sigma_sq)
    return w, mu, sigma_sq, sigma

cdef int K = 3
cdef float lambda_sq = 2.5**2
cdef float alpha = 0.2
cdef float T = 0.7
cdef int shape_0,shape_1,shape_2
 
# Testing
if __name__ == '__main__':
    vid = cv2.VideoCapture("./Test Data/2.avi")
    # vid = cv2.VideoCapture(0)
    ret, frame = vid.read()
    bgsub1 = cv2.createBackgroundSubtractorMOG2()
    bgsub2 = cv2.createBackgroundSubtractorKNN()

    # Initial values of parameters
    shape_0 = frame.shape[0]
    shape_1 = frame.shape[1]
    shape_2 = frame.shape[2]
    w = np.full((shape_0,shape_1,K),1/K,dtype=np.float64)
    mu = np.zeros((shape_0,shape_1,shape_2,K),dtype=np.float64)
    sigma = np.ones((shape_0,shape_1,K),dtype=np.float64)
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
        w, mu, sigma_sq, sigma = update(frame.astype(np.float64),gaussians,alpha,w,mu,sigma_sq,diff_sq_sum,update_mask,replace_mask)      

    vid.release()
    cv2.destroyAllWindows()