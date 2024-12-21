import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSampler(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = DoubleConv(in_dim, out_dim)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        d = self.conv(x)
        p = self.pool(d)
        return d, p

class UpSampler(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_dim, in_dim//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_dim, out_dim)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)