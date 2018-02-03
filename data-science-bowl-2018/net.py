"""
Made use of https://github.com/milesial/Pytorch-UNet/tree/master/unet
"""
import torch.nn as nn
import torch.nn.functional as F

# From end of section 2:
# To allow a seamless tiling of the output segmentation map (see Figure 2), it
# is important to select the input tile size such that all 2x2 max-pooling operations
# are applied to a layer with an even x- and y-size.
class u_net(nn.Module):
    def __init__(self):
        super(u_net, self).__init__()
        self.d1 = double_conv(1, 64)
        self.d2 = full_down(64, 128)
        self.d3 = full_down(128, 256)
        self.d4 = full_down(256, 512)
        self.d5 = full_down(512, 1024)

    def forward(self, *inp):
        x0 = inp[0]

        # These 4 downsamples are used in the up sample
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x = self.d5(x4)

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

    def forward(self, *inp):
        x = inp[0]
        return x

class test(nn.Module):
    def __init__(self):
        super(test, set).__init__()
        self.conv = nn.ConvTranspose1d(100, 100, 2)

    def forward(self, *inp):
        x = inp[0]
        return x
