"""
See the paper: https://arxiv.org/pdf/1505.04597.pdf
Made use of: https://github.com/milesial/Pytorch-UNet/tree/master/unet
"""
import torch.nn as nn
import torch
# import torch.nn.functional as F

# u_net predictive region is 184 shorter on each side than the input
# I think that will be hard for us. Also it is a massive network. I am going to drop 1 layer.
# now it is 88 (184 - 64 - 32) smaller.
# We will run a 348x348 input -> 260x260 output
class u_net(nn.Module):
    def __init__(self):
        super(u_net, self).__init__()

        self.inp = double_conv(1, 64)
        self.d1 = full_down(64, 128)
        self.d2 = full_down(128, 256)
        self.d3 = full_down(256, 512)

        # self.d4 = full_down(512, 1024)
        # self.u4 = full_up(1024, 512)

        self.u3 = full_up(512, 256)
        self.u2 = full_up(256, 128)
        self.u1 = full_up(128, 64)
        self.out = nn.Sequential(
                nn.Conv2d(64, 1, 1),
                # nn.ReLU(), This is a bad idea but I'm not sure why...
                # consider handtanh to put this between 0 and 1
        )

    def forward(self, *inp):
        x = inp[0]

        x1 = self.inp(x)

        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x = self.d3(x3)

        # x = self.d4(x4)
        # x = self.u4(x4, x)

        x = self.u3(x3, x)
        x = self.u2(x2, x)
        x = self.u1(x1, x)

        x = self.out(x)
        return x


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, *inp):
        x = inp[0]
        return self.conv(x)

class full_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(full_down, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2), # stride defaults to kernel_size
            double_conv(in_ch, out_ch),
        )

    def forward(self, *inp):
        x = inp[0]
        return self.conv(x)

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


##### Helpers
# def get_input_size_from_output(output_size):
#     x = output_size
#     dc = 4 # amount of size lost in double conv

#     up_steps, down_steps = 3, 3
#     for i in range(up_steps):
#         x += 4
#         x /= 2
#         if x % 2 == 1:
#             return 0
#     for i in range(down_steps):
#         x += 4
#         x *= 2
#     x += 4
#     return(x)
