#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn, optim
from torch.autograd import Variable
import pdb


class Discriminator(nn.Module):

    """
    multi-task CNN for identity and pose classification
    D是一个类似于encoder的卷积网络，用来提取特征，最后做id和pose的分类
    ### init
    Nd : Number of identitiy to classify
    Np : Number of pose to classify

    """

    def __init__(self, Nd, Np, channel_num):
        super(Discriminator, self).__init__()
        convLayers = [
            nn.Conv2d(channel_num, 32, 3, 1, 1, bias=False),  # Bxchx96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),  # Bx32x96x96 -> Bx64x96x96
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx64x96x96 -> Bx64x97x97
            nn.Conv2d(64, 64, 3, 2, 0, bias=False),  # Bx64x97x97 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # Bx64x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # Bx64x48x48 -> Bx128x48x48
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx128x48x48 -> Bx128x49x49
            nn.Conv2d(128, 128, 3, 2, 0, bias=False),  # Bx128x49x49 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False),  # Bx128x24x24 -> Bx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False),  # Bx96x24x24 -> Bx192x24x24
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx192x24x24 -> Bx192x25x25
            nn.Conv2d(192, 192, 3, 2, 0, bias=False),  # Bx192x25x25 -> Bx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False),  # Bx192x12x12 -> Bx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),  # Bx128x12x12 -> Bx256x12x12
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx256x12x12 -> Bx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False),  # Bx256x6x6 -> Bx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.Conv2d(160, 320, 3, 1, 1, bias=False),  # Bx160x6x6 -> Bx320x6x6
            nn.BatchNorm2d(320),
            nn.ELU(),
            nn.AvgPool2d(6, stride=1),  # Bx320x6x6 -> Bx320x1x1
        ]

        self.convLayers = nn.Sequential(*convLayers)
        self.fc = nn.Linear(320, Nd + 1 + Np)

        # 权重全部用N（0,0.02）初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    def forward(self, input):
        # 卷积->平均滤波结果获得Bx320x1x1的输出
        x = self.convLayers(input) # Tensor:(batch-size, 320, 1, 1)

        x = x.view(-1, 320) # Tensor:(batch-size, 320)

        # 全結合
        x = self.fc(x)  # Tensor:(batch-size, 320) -> Tensor:(batch-size, Nd+1+Np)

        return x # Tensor:(batch-size, Nd+1+Np)


## 即使继承了nn.Module，如果不在super中调用构造函数，则成员变量self._modues
## 在未定义的情况下进行后续权重初始化时出错
## self._模块是存储模块存储的模块名称的列表

class Crop(nn.Module):
    """
    Generator上采样时，与下采样时ZeroPad 2d相反的函数
    论文作者在Tensorflow中用padding='SAME'选项自动填充
    下采样时ZeroPad 2d，上采样时Crop实现
    ### init
    crop_列表：指定数据的上下左右各减少多少
    """

    def __init__(self, crop_list):
        super(Crop, self).__init__()

        # crop_lsit = [crop_top, crop_bottom, crop_left, crop_right]
        self.crop_list = crop_list

    def forward(self, x):
        B, C, H, W = x.size()
        x = x[:, :, self.crop_list[0]: H - self.crop_list[1], self.crop_list[2]: W - self.crop_list[3]]

        return x


def WSum_feature(x, n):
    """
    只对权重的输出部分施加信号函数，利用该权重对n张图像的输出结果进行相加
    输入：nBx321x1->输出：Bx320x1x1x1
    n:每个人有多少张图像作为数据传递
    B:每批都有多少人（1人n张）的图像作为数据传递

    """
    # nBx320x1x1 -> Bx320x1x1

    B = x.size(0) // n
    weight = x[:, -1].unsqueeze(1).sigmoid()
    features = x * weight
    features = features[:, :-1].split(n, 0)
    features = torch.cat(features, 1)
    features_summed = features.sum(0, keepdim=True)
    features_summed = features_summed.view(B, -1)

    return features_summed


class Generator(nn.Module):
    """
    Encoder/Decoder conditional GAN conditioned with pose vector and noise vector

    ### init
    Np : Dimension of pose vector (Corresponds to number of dicrete pose classes of the data)
    Nz : Dimension of noise vector
    n  : Number of images per person

    """

    def __init__(self, Np, Nz, channle_num, images_perID):
        super(Generator, self).__init__()
        self.features = []
        self.images_perID = images_perID

        G_enc_convLayers = [
            nn.Conv2d(channle_num, 32, 3, 1, 1, bias=False),  # nBxchx96x96 -> nBx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),  # nBx32x96x96 -> nBx64x96x96
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),  # nBx64x96x96 -> nBx64x97x97
            nn.Conv2d(64, 64, 3, 2, 0, bias=False),  # nBx64x97x97 -> nBx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # nBx64x48x48 -> nBx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # nBx64x48x48 -> nBx128x48x48
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),  # nBx128x48x48 -> nBx128x49x49
            nn.Conv2d(128, 128, 3, 2, 0, bias=False),  # nBx128x49x49 -> nBx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False),  # nBx128x24x24 -> nBx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False),  # nBx96x24x24 -> nBx192x24x24
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),  # nBx192x24x24 -> nBx192x25x25
            nn.Conv2d(192, 192, 3, 2, 0, bias=False),  # nBx192x25x25 -> nBx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False),  # nBx192x12x12 -> nBx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),  # nBx128x12x12 -> nBx256x12x12
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),  # nBx256x12x12 -> nBx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # nBx256x13x13 -> nBx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False),  # nBx256x6x6 -> nBx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),

            # 将表示将同一人物图像的特征量相加时的权重的值w追加到一维信道
            nn.Conv2d(160, 321, 3, 1, 1, bias=False),  # nBx160x6x6 -> nBx321x6x6
            nn.BatchNorm2d(321),
            nn.ELU(),
            nn.AvgPool2d(6, stride=1),  # nBx321x6x6 -> nBx321x1x1

        ]
        self.G_enc_convLayers = nn.Sequential(*G_enc_convLayers)

        G_dec_convLayers = [
            nn.ConvTranspose2d(320, 160, 3, 1, 1, bias=False),  # Bx320x6x6 -> Bx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.ConvTranspose2d(160, 256, 3, 1, 1, bias=False),  # Bx160x6x6 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 256, 3, 2, 0, bias=False),  # Bx256x6x6 -> Bx256x13x13
            nn.BatchNorm2d(256),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(256, 128, 3, 1, 1, bias=False),  # Bx256x12x12 -> Bx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 192, 3, 1, 1, bias=False),  # Bx128x12x12 -> Bx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ConvTranspose2d(192, 192, 3, 2, 0, bias=False),  # Bx128x12x12 -> Bx192x25x25
            nn.BatchNorm2d(192),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(192, 96, 3, 1, 1, bias=False),  # Bx192x24x24 -> Bx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.ConvTranspose2d(96, 128, 3, 1, 1, bias=False),  # Bx96x24x24 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 128, 3, 2, 0, bias=False),  # Bx128x24x24 -> Bx128x49x49
            nn.BatchNorm2d(128),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(128, 64, 3, 1, 1, bias=False),  # Bx128x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64, 3, 1, 1, bias=False),  # Bx64x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64, 3, 2, 0, bias=False),  # Bx64x48x48 -> Bx64x97x97
            nn.BatchNorm2d(64),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(64, 32, 3, 1, 1, bias=False),  # Bx64x96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, channle_num, 3, 1, 1, bias=False),  # Bx32x96x96 -> Bxchx96x96
            nn.Tanh(),
        ]

        self.G_dec_convLayers = nn.Sequential(*G_dec_convLayers)

        self.G_dec_fc = nn.Linear(320 + Np + Nz, 320 * 6 * 6)

        # 重みは全て N(0, 0.02) で初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    def forward(self, input, pose, noise, single=False):

        x = self.G_enc_convLayers(input)  # nBx1x96x96 -> Bx321x1x1

        x = x.view(-1, 321)

        if single:
            # 足し合わせない場合
            x = x[:, :-1]  # nBx321 -> nBx320
        else:
            # 使用权重对同一人物的图像的特征量进行相加
            x = WSum_feature(x, self.images_perID)  # nBx321 -> Bx320

        self.features = x

        x = torch.cat([x, pose, noise], 1)  # B(nB)x320 -> B(nB) x (320+Np+Nz)

        x = self.G_dec_fc(x)  # B(nB) x (320+Np+Nz) -> B(nB) x (320x6x6)

        x = x.view(-1, 320, 6, 6)  # B(nB) x (320x6x6) -> B(nB) x 320 x 6 x 6

        x = self.G_dec_convLayers(x)  # B(nB) x 320 x 6 x 6 -> B(nB)x1x96x96

        return x
