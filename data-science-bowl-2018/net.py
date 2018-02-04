"""
See the paper: https://arxiv.org/pdf/1505.04597.pdf
Made use of: https://github.com/milesial/Pytorch-UNet/tree/master/unet
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class u_net(nn.Module):
    def __init__(self):
        super(u_net, self).__init__()
        self.d1 = double_conv(1, 64)
        self.d2 = full_down(64, 128)
        self.d3 = full_down(128, 256)
        self.d4 = full_down(256, 512)
        self.d5 = full_down(512, 1024)

        self.u1 = full_up(1024, 512)
        self.u2 = full_up(512, 256)
        self.u3 = full_up(256, 128)
        self.u4 = full_up(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, *inp):
        x = inp[0]

        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x = self.d5(x4)

        x = self.u1(x4, x)
        x = self.u2(x3, x)
        x = self.u3(x2, x)
        x = self.u4(x1, x)

        x = self.out(x)
        return x


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, *inp):
        x = inp[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class full_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(full_down, self).__init__()
        self.maxpool = nn.MaxPool2d(2) # stride defaults to kernel_size
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, *inp):
        x = inp[0]
        x = self.maxpool(x)
        x = self.conv(x)
        return x

class full_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(full_up, self).__init__()
        # Slightly confusing. We transpose it to the number of out channels
        # Then we double the number of channels (pulling in data from across the U)
        # Then we halve the number of channels with the convolutions
        self.transpose_conv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = double_conv(2*out_ch, out_ch)

    def forward(self, *inp):
        across, x = inp[0], inp[1]
        x = self.transpose_conv(x)

        s = int((across.shape[3] - x.shape[3])/2)
        across = across[:,:,s:-s,s:-s]
        # pylint: disable=E1101
        x = torch.cat((across, x), dim = 1)
        x = self.conv(x)
        return x
