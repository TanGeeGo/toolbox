import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchstat import stat # complexity evaluation

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicBlock, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicBlock(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicBlock(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicBlock(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicBlock(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)

class MFF(nn.Module):
    def __init__(self, scale, out_channel, base_channel=64):
        super(MFF, self).__init__()
        self.conv_shuffle = nn.ModuleList([
            BasicBlock(base_channel, base_channel, kernel_size=7, stride=1, relu=True),
            BasicBlock(base_channel*2, base_channel*4, kernel_size=5, stride=1, relu=True),
            BasicBlock(base_channel*4, base_channel*16, kernel_size=3, stride=1, relu=True),
        ])

        self.shuffle = nn.ModuleList([
            nn.PixelShuffle(1) if scale==1 else nn.PixelUnshuffle(scale),
            nn.PixelShuffle(int(2/scale)),
            nn.PixelShuffle(int(4/scale))
        ])

        self.conv_out = nn.Sequential(
            BasicBlock(3*(scale**2)*base_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x1_ = self.shuffle[0](self.conv_shuffle[0](x1))
        x2_ = self.shuffle[1](self.conv_shuffle[1](x2))
        x4_ = self.shuffle[2](self.conv_shuffle[2](x4))

        x = torch.cat([x1_, x2_, x4_], dim=1)
        return self.conv_out(x)

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicBlock(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicBlock(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicBlock(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicBlock(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicBlock(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicBlock(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class MIMOUNet(nn.Module):
    def __init__(self, num_res=8, base_channel=32):
        super(MIMOUNet, self).__init__()

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicBlock(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicBlock(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicBlock(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicBlock(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicBlock(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicBlock(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicBlock(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicBlock(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicBlock(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicBlock(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


class MIMOUNetPlus(nn.Module):
    def __init__(self, num_res=20, base_channel=32):
        super(MIMOUNetPlus, self).__init__()

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicBlock(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicBlock(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicBlock(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicBlock(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicBlock(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicBlock(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicBlock(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicBlock(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicBlock(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicBlock(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs

class MIMOUNetMFF(nn.Module):
    def __init__(self, num_res=8, base_channel=32):
        super(MIMOUNetMFF, self).__init__()

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicBlock(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicBlock(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicBlock(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicBlock(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicBlock(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicBlock(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicBlock(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicBlock(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicBlock(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicBlock(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.MFFs = nn.ModuleList(
            [
                MFF(scale=1, out_channel=base_channel, base_channel=base_channel),
                MFF(scale=2, out_channel=base_channel*2, base_channel=base_channel)
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        res3 = self.Encoder[2](z)

        res1_ = self.MFFs[0](res1, res2, res3)
        res2_ = self.MFFs[1](res1, res2, res3)

        z = self.Decoder[0](res3)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2_], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1_], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs

# def build_net(model_name):
#     class ModelError(Exception):
#         def __init__(self, msg):
#             self.msg = msg

#         def __str__(self):
#             return self.msg

#     if model_name == "MIMO-UNetPlus":
#         return MIMOUNetPlus()
#     elif model_name == "MIMO-UNet":
#         return MIMOUNet()
#     elif model_name == "MIMO-UNet-MFF":
#         return MIMOUNetMFF()
#     raise ModelError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')

def get_parameter_number(model, input_size=(3, 224, 224)):
    stat(model, input_size)

if __name__ == '__main__':
    model = MIMOUNet()
    get_parameter_number(model)