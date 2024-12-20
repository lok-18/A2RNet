import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import math


class Convdown(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        self.convd = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim, 1, 1, 0))

        self.attn = ESSAttn(dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        shortcut = x
        x_size = (x.shape[2], x.shape[3])
        x_embed = self.patch_embed(x)
        x_embed = self.attn(self.norm(x_embed))  # + x_embed
        x = self.drop(self.patch_unembed(x_embed, x_size))
        x = torch.cat((x, shortcut), dim=1)
        x = self.convd(x)
        x = x + shortcut
        return x


class Convup(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        self.convu = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim, 1, 1, 0))
        self.drop = nn.Dropout2d(0.2)
        self.norm = nn.LayerNorm(dim)
        self.attn = ESSAttn(dim)

    def forward(self, x):
        shortcut = x
        x_size = (x.shape[2], x.shape[3])
        x_embed = self.patch_embed(x)
        x_embed = self.attn(self.norm(x_embed))
        x = self.drop(self.patch_unembed(x_embed, x_size))
        x = torch.cat((x, shortcut), dim=1)
        x = self.convu(x)
        x = x + shortcut
        return x


class blockup(nn.Module):
    def __init__(self, dim, upscale):
        super(blockup, self).__init__()
        self.convup = Convup(dim)
        self.convdown = Convdown(dim)
        self.convupsample = Upsample(scale=upscale, num_feat=dim)
        self.convdownsample = Downsample(scale=upscale, num_feat=dim)

    def forward(self, x):
        xup = self.convupsample(x)
        x1 = self.convup(xup)
        xdown = self.convdownsample(x1) + x
        x2 = self.convdown(xdown)
        xup = self.convupsample(x2) + x1
        x3 = self.convup(xup)
        xdown = self.convdownsample(x3) + x2
        x4 = self.convdown(xdown)
        xup = self.convupsample(x4) + x3
        x5 = self.convup(xup)
        return x5


class PatchEmbed(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x

#ARB
class ESSAttn(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        b, N, C = x.shape
        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)
        # t2 = self.norm1(t2)*0.3
        # print(torch.mean(t1),torch.std(t1))
        # print(torch.mean(t2), torch.std(t2))
        # t2 = self.norm1(t2)*0.1
        # t2 = ((q2 / (q2s+1e-7)) @ t2)

        # q3 = torch.pow(q,4)
        # q3s = torch.pow(q2s,2)
        # k3 = torch.pow(k, 4)
        # k3s = torch.sum(k2,dim=2).unsqueeze(2).repeat(1, 1, C)
        # t3 = ((k3 / k3s)*16).transpose(-2, -1) @ v
        # t3 = ((q3 / q3s)*16) @ t3
        # print(torch.max(t1))
        # print(torch.max(t2))
        # t3 = (((torch.pow(q,4))/24) @ (((torch.pow(k,4).transpose(-2,-1))/24)@v)*16/math.sqrt(N))
        attn = t1 + t2
        attn = self.ln(attn)
        return attn

    def is_same_matrix(self, m1, m2):
        rows, cols = len(m1), len(m1[0])
        for i in range(rows):
            for j in range(cols):
                if m1[i][j] != m2[i][j]:
                    return False
        return True


class Downsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1))
                m.append(nn.PixelUnshuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, num_feat // 9, 3, 1, 1))
            m.append(nn.PixelUnshuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Downsample, self).__init__(*m)


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

# DRM
class ESSA(nn.Module):
    def __init__(self, inch, dim, upscale, **kwargs):
        super(ESSA, self).__init__()
        self.conv_first = nn.Conv2d(inch, dim, 3, 1, 1)
        self.blockup = blockup(dim=dim, upscale=upscale)
        self.conv_last = nn.Conv2d(dim, inch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.blockup(x)
        x = self.conv_last(x)
        return x


class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.Dropout(0.1), 
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)

class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.MaxPool2d(2)

    def forward(self, x):
        return self.Down(x)

class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        return torch.cat((x, r), 1)

# A2RNet
class ESSA_UNet(nn.Module):

    def __init__(self):
        super(ESSA_UNet, self).__init__()
        self.C1 = Conv(2, 16)

        self.C1_ESSA = ESSA(16, 32, 1)
        self.C2_ESSA = ESSA(32, 64, 1)
        self.C3_ESSA = ESSA(64, 128, 1)
        self.C4_ESSA = ESSA(128, 256, 1)

        self.D1 = DownSampling(16)
        self.C2 = Conv(16, 32)
        self.D2 = DownSampling(32)
        self.C3 = Conv(32, 64)
        self.D3 = DownSampling(64)
        self.C4 = Conv(64, 128)
        self.D4 = DownSampling(128)
        self.C5 = Conv(128, 256)

        self.U1 = UpSampling(256)
        self.C6 = Conv(256, 128)
        self.U2 = UpSampling(128)
        self.C7 = Conv(128, 64)
        self.U3 = UpSampling(64)
        self.C8 = Conv(64, 32)
        self.U4 = UpSampling(32)
        self.C9 = Conv(32, 16)

        self.Th = nn.Sigmoid()
        self.pred = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x1, x2):  # x1:vis  x2:ir
        x1 = x1[:, : 1]
        x = torch.cat((x1, x2), dim=1) 
        
        R1 = self.C1(x)
        R1 = self.C1_ESSA(R1)
        
        R2 = self.C2(self.D1(R1))
        
        R3 = self.C3(self.D2(R2))
        R3 = self.C3_ESSA(R3)
        
        R4 = self.C4(self.D3(R3))

        Y1 = self.C5(self.D4(R4))

        O1 = self.C6(self.U1(Y1, R4))

        O2 = self.C7(self.U2(O1, R3))

        O3 = self.C8(self.U3(O2, R2))

        O4 = self.C9(self.U4(O3, R1))
        
        return self.Th(self.pred(O4))
