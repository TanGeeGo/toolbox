"""
The frequency self-adaptive network for optical degradation correction
paper link:
@article{Lin_2022_OE,
  author = {Ting Lin and ShiQi Chen and Huajun Feng and Zhihai Xu and Qi Li and Yueting Chen},
  journal = {Opt. Express},
  keywords = {All optical devices; Blind deconvolution; Image processing; Image quality; Optical design; Ray tracing},
  number = {13},
  pages = {23485--23498},
  publisher = {Optica Publishing Group},
  title = {Non-blind optical degradation correction via frequency self-adaptive and finetune tactics},
  volume = {30},
  month = {Jun},
  year = {2022},
  url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-30-13-23485},
  doi = {10.1364/OE.458530},
  abstract = {In mobile photography applications, limited volume constraints the diversity of optical design. In addition to the narrow space, the deviations introduced in mass production cause random bias to the real camera. In consequence, these factors introduce spatially varying aberration and stochastic degradation into the physical formation of an image. Many existing methods obtain excellent performance on one specific device but are not able to quickly adapt to mass production. To address this issue, we propose a frequency self-adaptive model to restore realistic features of the latent image. The restoration is mainly performed in the Fourier domain and two attention mechanisms are introduced to match the feature between Fourier and spatial domain. Our method applies a lightweight network, without requiring modification when the fields of view (FoV) changes. Considering the manufacturing deviations of a specific camera, we first pre-train a simulation-based model, then finetune it with additional manufacturing error, which greatly decreases the time and computational overhead consumption in implementation. Extensive results verify the promising applications of our technique for being integrated with the existing post-processing systems.},
}
"""
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils_filter import eptional_fft, kernel_fft_t, gauss_fit

#-------------------------------------------------------
# network utils
#-------------------------------------------------------

def kernel_norm(kernel_set):
    """
    normalize the kernel to sum 1
    """
    kernel_set_norm = torch.zeros_like(kernel_set).to(kernel_set.device)
    for i in range(kernel_set.shape[0]):
        kernel_set_norm[i, ...] = kernel_set[i, ...] / kernel_set[i, ...].sum()
    
    return kernel_set_norm

# FFT attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes*8, 1, bias=False), # used to in_planes//16 
                               nn.ReLU(),
                               nn.Conv2d(in_planes *8, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class kernel_conv(nn.Module):
    """
    self-defined convolution for attention machanism in frequency domain 
    """
    def __init__(self, input_channel=96, output_channel=96, kernel_num=5, sig_num=25):
        super(kernel_conv,self).__init__()
        
        self.ca = ChannelAttention(sig_num)
        self.sa = SpatialAttention()
        self.eptional_att = nn.Conv2d(input_channel, sig_num, 3, 1, 1) # for high frequency
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.ep_ker_cha = nn.Conv2d(sig_num, kernel_num, 3, 1, 1)

        # the channel of eptional accords to the kernel_num, so here every kernel has its eptional
        self.half_cha = nn.Conv2d(input_channel*kernel_num, output_channel, 3, 1, 1)

    def forward(self, x, eptional_map, kernel_set):

        #frequency attention kernel_set [5,17,17]
        fft_x = torch.abs(torch.fft.fft2(x))
        
        eptional_att=self.lrelu(self.eptional_att(fft_x))
        # filter fit
        eptional_map_fft = eptional_fft(eptional_map, x.shape) # [5, h, w]
        eptional = eptional_att * torch.abs(eptional_map_fft) * torch.abs(eptional_map_fft) # [B, 5, h, w]
        # -----------------------------------------------------------------
        # make less difference in ablation
        # eptional = self.ca(eptional)*eptional # map attention
        # eptional = self.sa(eptional)*eptional # map attention
        # -----------------------------------------------------------------
        eptional =self.lrelu(self.ep_ker_cha(eptional))# [B, ker_num, h, w]

        # kernel set
        kernel = kernel_set.unsqueeze(0).expand(eptional.shape[0], -1, -1, -1) # [B, 5, h, w]
        k_fft = kernel_fft_t(kernel, x.shape, eptional) # [B, 5, h, w]
        
        out = list()
        for i in range(k_fft.shape[1]):
            temp = fft_x*k_fft[:,i,:,:].unsqueeze(1)
            temp = torch.abs(torch.fft.ifft2(temp))
            out.append(temp)

        out = torch.cat(out, dim=1)

        return self.lrelu(self.half_cha(out))
    
# scale fliter attention
class scale_att(nn.Module):
    def __init__(self,input_channel=96,out_channel=48):
        super(scale_att,self).__init__()

        self.ac1=nn.Conv2d(input_channel,32,kernel_size=5,dilation=2,padding=4)
        self.ac2=nn.Conv2d(32,32,kernel_size=5,dilation=2,padding=4)
        self.ac3=nn.Conv2d(32,16,kernel_size=5,dilation=2,padding=4)
        self.ac4=nn.Conv2d(16,16,kernel_size=5,dilation=2,padding=4)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2,inplace=False)
        self.ac5 = nn.Conv2d(16,out_channel,kernel_size= 5,padding = 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        ac1 = self.lrelu(self.ac1(x))
        ac2 = self.lrelu(self.ac2(ac1))
        ac3 = self.lrelu(self.ac3(ac2))
        ac4 = self.lrelu(self.ac4(ac3))
        ac5 = self.sigmoid(self.ac5(ac4))
        
        return ac5
    
class channel_att(nn.Module):
    def __init__(self,input_channel=96,out_channel=48):
        super(channel_att,self).__init__()
        self.conv1 = nn.Conv2d(input_channel,int(input_channel/4),1,1,0)
        self.conv2 = nn.Conv2d(int(input_channel/4),out_channel,1,1,0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2,inplace=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = F.adaptive_avg_pool2d(x,(1,1))
        x = self.lrelu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        return x

class KPAC(nn.Module):

    def __init__(self, input_channel=96, out_channel=96, kernel_num=5, sig_num=25):
        super(KPAC,self).__init__()
        # scale attention module
        self.att1 = scale_att(input_channel=input_channel,out_channel=kernel_num)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2,inplace=False)
        self.sigmoid = nn.Sigmoid()

        # shape attention module 
        self.cha_att = channel_att(input_channel=96,out_channel=int(input_channel/2))

        self.ac5 = nn.Conv2d(int(input_channel*kernel_num/2), int(input_channel/2),3,1,1)
        self.kernel_conv = kernel_conv(input_channel, int(out_channel/2)*kernel_num, kernel_num, sig_num)

        # fusion
        self.conv2 = nn.Conv2d(int(input_channel/2)*kernel_num,out_channel,kernel_size=3,stride=1,padding=1) 

    # input of the model is image and kernel (dict)
    def forward(self, x, eptional_map, kernel_set):
        
        alpha = self.att1(x)
        beta = self.cha_att(x)
        x_K1 = self.kernel_conv(x, eptional_map, kernel_set)
    
        m = int(x_K1.shape[1] / alpha.shape[1])
        out = list()
        for i in range(alpha.shape[1]):
            temp = x_K1[:,i*m:(i+1)*m,:,:] * alpha[:,i,:,:].unsqueeze(1)*beta
            out.append(temp)
            
        out=torch.cat(out, dim=1)
        # fusion
        out =self.lrelu(self.conv2(out))
        return out

class FSANet(nn.Module):
    def __init__(self, 
                 input_channel=5, 
                 output_channel=3, 
                 base_channel=48, 
                 kernel_num=25, 
                 sig_num=25):
        super(FSANet,self).__init__()
        
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.base_channel = base_channel
        
        #encoder
        self.conv1_1 = nn.Conv2d(self.input_channel, self.base_channel, kernel_size=5, stride=1, padding=2)
        self.conv1_2 = nn.Conv2d(self.base_channel, self.base_channel, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(self.base_channel, self.base_channel, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(self.base_channel, self.base_channel, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(self.base_channel, self.base_channel*2, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(self.base_channel*2, self.base_channel*2, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(self.base_channel*2, self.base_channel*2, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(self.base_channel*2, self.base_channel*2, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2,inplace=True)

        # kernel downsample
        
        self.kconv2_1 = nn.Conv2d(kernel_num, kernel_num, kernel_size=3, stride=2, padding=1)
        self.kconv3_1 = nn.Conv2d(kernel_num, kernel_num, kernel_size=3, stride=2, padding=1)
        self.kconv4_1 = nn.Conv2d(kernel_num, kernel_num, kernel_size=3, stride=2, padding=1)
        #KPAC block
        self.kpac4 = KPAC(input_channel=self.base_channel*2, out_channel=self.base_channel*2, 
                          kernel_num=kernel_num, sig_num=sig_num)

        #decoder
        self.conv5_1 = nn.Conv2d(self.base_channel*4, self.base_channel*2, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(self.base_channel*2, self.base_channel*2, kernel_size=3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(self.base_channel*2, self.base_channel*2, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(self.base_channel*4, self.base_channel*2, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(self.base_channel*2, self.base_channel, kernel_size=4, stride=2, padding=1)
        self.conv7 = nn.Conv2d(self.base_channel*2, self.base_channel, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(self.base_channel, self.base_channel, kernel_size=4, stride=2, padding=1)
        self.conv8 = nn.Conv2d(self.base_channel*2, self.output_channel, kernel_size=5, stride=1, padding=2)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                # assert m.kernel_size[0] == m.kernel_size[1]
                # m.weight.data = nn.init.kaiming_normal(m.weight.data)
    
    def forward(self, x, kernel_set, eptional_map):
        # 1st down sampling
        conv1_1 = self.lrelu(self.conv1_1(x))
        conv1_2 = self.lrelu(self.conv1_2(conv1_1))

        # 2nd downsample
        conv2_1 = self.lrelu(self.conv2_1(conv1_2))
        conv2_2 = self.lrelu(self.conv2_2(conv2_1))
        kernel2_2 = self.lrelu(self.kconv2_1(kernel_set[0, ...].unsqueeze(0))).squeeze(0)
        kernel2_2 = kernel_norm(kernel2_2)

        #3rd downsample
        conv3_1 = self.lrelu(self.conv3_1(conv2_2))
        conv3_2 = self.lrelu(self.conv3_2(conv3_1))
        kernel3_2 = self.lrelu(self.kconv3_1(kernel2_2.unsqueeze(0))).squeeze(0)
        kernel3_2 = kernel_norm(kernel3_2)

        #4th downsample
        conv4_1 = self.lrelu(self.conv4_1(conv3_2))
        conv4_2 = self.lrelu(self.conv4_2(conv4_1))
        kernel4_2 = self.lrelu(self.kconv4_1(kernel3_2.unsqueeze(0))).squeeze(0)
        kernel4_2 = kernel_norm(kernel4_2)
        kpac1 = self.kpac4(conv4_2, eptional_map[0, ...], kernel4_2)
        cat4 = torch.cat([conv4_2, kpac1], dim =1)

        conv5_1 = self.lrelu(self.conv5_1(cat4))
        conv5_2 = self.lrelu(self.conv5_2(conv5_1))
        deconv1 = self.lrelu(self.deconv1(conv5_2))
        cat3 = torch.cat([deconv1,conv3_2],dim = 1)
        conv6 = self.lrelu(self.conv6(cat3))
        deconv2 = self.lrelu(self.deconv2(conv6))
        cat2 = torch.cat([deconv2,conv2_2],dim = 1)

        conv7 = self.lrelu(self.conv7(cat2))
        deconv3 = self.lrelu(self.deconv3(conv7))
        cat1 = torch.cat([deconv3,conv1_2],dim = 1)
        conv8 = self.lrelu(self.conv8(cat1))
        
        # res
        y=conv8+x[:,0:3,:,:]

        return y

if __name__ == "__main__":
    net = FSANet()
    kernel_set = torch.randn(5, 31, 31)
    eptional_map = torch.FloatTensor(gauss_fit(100, 100, 0.2, 2.5, 25))
    image = torch.randn(8, 5, 200, 200)

    out = net(image, kernel_set, eptional_map)
    print(out.shape)
