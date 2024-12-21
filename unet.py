import torch
import torch.nn as nn

from elements import *

class UNet(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.down_conv1 = DownSampler(in_dim, 64)
        self.down_conv2 = DownSampler(64, 128)
        self.down_conv3 = DownSampler(128, 256)
        self.down_conv4 = DownSampler(256, 512)

        self.bridge = DoubleConv(512, 1024)

        self.up_conv1 = UpSampler(1024, 512)
        self.up_conv2 = UpSampler(512, 256)
        self.up_conv3 = UpSampler(256, 128)
        self.up_conv4 = UpSampler(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        d1, p1 = self.down_conv1(x)
        d2, p2 = self.down_conv2(p1)
        d3, p3 = self.down_conv3(p2)
        d4, p4 = self.down_conv4(p3)

        b = self.bridge(p4)

        u1 = self.up_conv1(b, d4)
        u2 = self.up_conv2(u1, d3)
        u3 = self.up_conv3(u2, d2)
        u4 = self.up_conv4(u3, d1)

        out = self.out(u4)
        return out