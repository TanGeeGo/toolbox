import cv2
import torch
import numpy as np

#-------------------------------------------------------
# filter utils
#-------------------------------------------------------

#define 2d gaussian kernel
def gaussian_kernel_2d(ksize, sigma):

    return cv2.getGaussianKernel(ksize,sigma) * np.transpose(cv2.getGaussianKernel(ksize, sigma))

# kernel wiener filter inverse
def kernel_inv(kernel):

    fft = np.fft.fft2(kernel)
    k_inv = np.fft.ifft2(np.conj(fft) / (np.abs(fft)*np.abs(fft)+1e-2))
    
    return np.abs(k_inv) / np.sum(np.abs(k_inv))

# generate inverse kernel of different sigma
def gen_gausskernel_ivs(ksize, sigma_range):

    k_ivs=np.zeros((len(sigma_range), ksize, ksize))
    for i in range(len(sigma_range)):

        temp=gaussian_kernel_2d(ksize, sigma_range[i])
        k_ivs[i, :, :]=kernel_inv(temp)
    
    return k_ivs

# kernel filter fft2 like wiener filter
def kernel_fft(kernel, patch_size, eptional):

    # generate fft kernel size the same as img size
    fft = np.fft.fft2(kernel, (patch_size, patch_size))
    k_size = kernel.shape[-1]
    k_fft = np.zeros((k_size, k_size), dtype=complex)
    k_fft = np.conj(fft) / (np.abs(fft) * np.abs(fft) + eptional)
    
    return k_fft

def kernel_fft_t(kernel, x_shape, eptional):

    fft = torch.fft.fft2(kernel, (x_shape[-2],x_shape[-1])) # generate fft kernel size the same as img size
    
    k_fft = torch.conj(fft)/(torch.abs(fft)*torch.abs(fft)+eptional)
    
    return k_fft

def eptional_fft(eptional_map, x_shape):

    # calculate eptional fft /size fit different input scale
    eptional_map_fft = list()
    for i in range(eptional_map.shape[0]):
        eptional_map_fft.append(torch.fft.fft2(eptional_map[i,:,:], (x_shape[-2], x_shape[-1])).unsqueeze(0))

    eptional_map_fft.append(torch.ones_like(eptional_map_fft[0]).to(eptional_map.device))
    
    return torch.cat(eptional_map_fft, dim=0)

# leastSquare eptional
def leastSquare(patch_size):
    #laplace operator
    la = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    la_fft = np.fft.fft2(la,(patch_size,patch_size))

    return la_fft

def sinefit(H, W, omega_num=10, theta_num=20, sigma=2):
    """
    fit the epsilon with sine wave in case of the missing of epsilon 
    """
    print('sine fit,omega_num:{},theta_num:{},sigma={}\n'.format(omega_num,theta_num,sigma))
    #low frequency,generate gaussian kernel
    low_mat = gaussian_kernel_2d(H,sigma)
    low_mat_f = np.fft.fft2(low_mat)

    # middle frequency, omega range[0,1),default omega number=10,default theta_num = 20

    omega = np.random.uniform(0.8,0.9,omega_num)
    theta = 360*np.random.uniform(size=theta_num)
    mid_mat = np.zeros((H,W))
    mid_mat_f = np.zeros((omega_num*theta_num,H,W))
    # fit with the sine wave of different omegas and thetas
    for i in range(omega_num):
        for j in range(theta_num):

            w1 = np.sin(theta[j])
            w2 = np.cos(theta[j])
            
            # meshgrid
            h = np.linspace(1,H,H)
            w = np.linspace(1,W,W)
            w_mat,h_mat = np.meshgrid(w,h)
            mid_mat= np.cos(omega[i]*(w1*h_mat+w2*w_mat))
            mid_mat_f[i*theta_num+j,:,:]=np.fft.fft2(mid_mat)

    # high frequency,import gen_leastSquare_fft to generate laplace matrix
    la = [[0,-1,0],[-1,4,-1],[0,-1,0]]
    high_mat_f = np.fft.fft2(la,(H,W))

    return low_mat_f, mid_mat_f, high_mat_f # return the frequency of low, middle, and high

def gauss_fit(H, W, sig_low=0.2, sig_high=2.5, sig_num=25):
    """
    fit the epsilon with gaussian kernel in case of the missing of epsilon 
    """
    sig_interval = (sig_high-sig_low-0.001)/(sig_num-1)
    eption_map = np.zeros((sig_num,H,W))
    for i in range(sig_num-1):

        sigma = sig_low + sig_interval*i
        eption_map[i,:,:] = gaussian_kernel_2d(H,sigma)
        # eption_map[i,:,:] = np.fft.fft2(eption_map[i,:,:])
    eption_map[sig_num-1,:,:] = np.ones((H,W))

    return eption_map

