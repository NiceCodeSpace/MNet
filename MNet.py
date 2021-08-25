import torch
from torch import nn
import torch.nn.functional as F

class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kSize, padding, if_norm=True, if_activation=True):
        super().__init__()
        self.if_norm = if_norm
        self.if_activation = if_activation

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kSize, stride=1, padding=padding)
        if self.if_norm:
            self.norm = nn.InstanceNorm3d(out_channels)

        if self.if_activation:
            self.prelu = nn.PReLU()

    def forward(self, x):
        o = self.conv(x)
        if self.if_norm:
            o = self.norm(o)
        if self.if_activation:
            o = self.prelu(o)
        return o


class convBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kSize, padding, if_norm=(True, True), if_activation=(True, True)):
        super().__init__()
        self.conv1 = conv(in_channels, out_channels, kSize, padding, if_norm[0], if_activation[0])
        self.conv2 = conv(out_channels, out_channels, kSize, padding, if_norm[1], if_activation[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class down(nn.Module):
    def __init__(self, in_channels, out_channels, minsize_to_pool):
        super().__init__()
        self.minsize_to_pool = minsize_to_pool
        self.in_channels = in_channels

        self.convBlock2d = convBlock(self.in_channels, out_channels, (1, 3, 3), padding=(0, 1, 1))
        self.convBlock3d = convBlock(self.in_channels, out_channels, (3, 3, 3), padding=(1, 1, 1))

        self.pool2d = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        if type(x) == type([]):  # x can be a tensor and a list of tensor
            x = torch.cat(x, dim=1)
        # 2d
        conv2d = self.convBlock2d(x)
        pool2d = self.pool2d(conv2d)
        # 3d
        conv3d = self.convBlock3d(x)
        if conv3d.shape[2] >= self.minsize_to_pool:
            pool3d = F.max_pool3d(conv3d, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        else:
            pool3d = F.max_pool3d(conv3d, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        return conv2d, pool2d, conv3d, pool3d


class up(nn.Module):
    def __init__(self, in_channels, out_channels, mode):
        super().__init__()
        self.mode = mode
        if mode == '2d':
            self.convBlock = convBlock(in_channels, out_channels, (1, 3, 3), padding=(0, 1, 1))
        elif mode == '3d':
            self.convBlock = convBlock(in_channels, out_channels, (3, 3, 3), padding=(1, 1, 1))
        elif mode == 'both':
            self.convBlock2d = convBlock(in_channels, out_channels, (1, 3, 3), padding=(0, 1, 1))
            self.convBlock3d = convBlock(in_channels, out_channels, (3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        x2d, xskip2d, x3d, xskip3d = x
        tarSize = xskip2d.shape[2:]
        up2d = F.interpolate(x2d, size=tarSize, mode='trilinear', align_corners=False)
        up3d = F.interpolate(x3d, size=tarSize, mode='trilinear', align_corners=False)

        cat = torch.cat([up2d, xskip2d, up3d, xskip3d], dim=1)

        if self.mode == 'both':
            conv2d = self.convBlock2d(cat)
            conv3d = self.convBlock3d(cat)
            return conv2d, conv3d
        else:
            return self.convBlock(cat)


class bottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, mode):  # mode:2d,3d,both
        super().__init__()
        self.mode = mode
        if mode == '2d':
            self.conv = convBlock(in_channels, out_channels, (1, 3, 3), padding=(0, 1, 1))

        elif mode == '3d':
            self.conv = convBlock(in_channels, out_channels, (3, 3, 3), padding=(1, 1, 1))

        elif mode == 'both':
            self.conv2d = convBlock(in_channels, out_channels, (1, 3, 3), padding=(0, 1, 1))
            self.conv3d = convBlock(in_channels, out_channels, (3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        if self.mode == 'both':
            x = torch.cat(x, dim=1)
            conv2d = self.conv2d(x)
            conv3d = self.conv3d(x)
            return conv2d, conv3d
        else:
            conv = self.conv(x)
            return conv


class MNet(nn.Module):
    def __init__(self, num_classes, channels=(48,48,48,48,48),minsize_to_pool=8,if_ds=True):
        super().__init__()
        self.num_classes = num_classes
        self.mtp = minsize_to_pool  # if size of axis z is lower than mtp, pooling size:3*3*3->1*3*3
        self.if_ds = if_ds # deep supervision

        self.down11 = down(1, channels[0], self.mtp)
        self.down12 = down(channels[0], channels[1], self.mtp)
        self.down13 = down(channels[1], channels[2], self.mtp)
        self.down14 = down(channels[2], channels[3], self.mtp)
        self.bottleneck1 = bottleNeck(channels[3], channels[4], mode='2d')
        self.up11 = up(2*channels[3]+2*channels[4], channels[3], '2d')
        self.up12 = up(2*channels[2]+2*channels[3], channels[2], '2d')
        self.up13 = up(2*channels[1]+2*channels[2], channels[1], '2d')
        self.up14 = up(2*channels[0]+2*channels[1], channels[0], 'both')#!!!!!!!!!

        self.down21 = down(channels[0], channels[1], self.mtp)
        self.down22 = down(channels[1] * 2, channels[2], self.mtp)
        self.down23 = down(channels[2] * 2, channels[3], self.mtp)
        self.bottleneck2 = bottleNeck(channels[3] * 2, channels[4], mode='both')
        self.up21 = up(2*channels[3]+2*channels[4], channels[3], 'both')
        self.up22 = up(2*channels[2]+2*channels[3], channels[2], 'both')
        self.up23 = up(2*channels[1]+2*channels[2], channels[1], '3d')

        self.down31 = down(channels[1], channels[2], self.mtp)
        self.down32 = down(channels[2] * 2, channels[3], self.mtp)
        self.bottleneck3 = bottleNeck(channels[3] * 2, channels[4], mode='both')
        self.up31 = up(2*channels[3]+2*channels[4], channels[3], mode='both')
        self.up32 = up(2*channels[2]+2*channels[3], channels[2], mode='3d')

        self.down41 = down(channels[2], channels[3], self.mtp)
        self.bottleneck4 = bottleNeck(channels[3]*2, channels[4], mode='both')
        self.up41 = up(2*channels[3]+2*channels[4], channels[3], mode='3d')

        self.bottleneck5 = bottleNeck(channels[3], channels[4], mode='3d')

        self.outputs = nn.ModuleList(
            [nn.Conv3d(channels[0] * 2, self.num_classes, kernel_size=(1, 1, 1), stride=1, padding=0)] +
            [nn.Conv3d(c, self.num_classes, kernel_size=(1, 1, 1), stride=1, padding=0)
             for c in [channels[1],channels[1],channels[2],channels[2],channels[3],channels[3]]]
            )

    def forward(self, x):
        down11 = self.down11(x)
        down12 = self.down12(down11[1])
        down13 = self.down13(down12[1])
        down14 = self.down14(down13[1])
        bottleNeck1 = self.bottleneck1(down14[1])

        down21 = self.down21(down11[3])
        down22 = self.down22([down21[1], down12[3]])
        down23 = self.down23([down22[1], down13[3]])
        bottleNeck2 = self.bottleneck2([down23[1], down14[3]])

        down31 = self.down31(down21[3])
        down32 = self.down32([down31[1], down22[3]])
        bottleNeck3 = self.bottleneck3([down32[1], down23[3]])

        down41 = self.down41(down31[3])
        bottleNeck4 = self.bottleneck4([down41[1], down32[3]])


        bottleNeck5 = self.bottleneck5(down41[3])


        up41 = self.up41([bottleNeck4[0], down41[0], bottleNeck5, down41[2]])

        up31 = self.up31([bottleNeck3[0], down32[0], bottleNeck4[1], down32[2]])
        up32 = self.up32([up31[0], down31[0], up41, down31[2]])

        up21 = self.up21([bottleNeck2[0], down23[0], bottleNeck3[1], down23[2]])
        up22 = self.up22([up21[0], down22[0], up31[1], down22[2]])
        up23 = self.up23([up22[0], down21[0], up32, down21[2]])


        up11 = self.up11([bottleNeck1, down14[0], bottleNeck2[1], down14[2]])
        up12 = self.up12([up11, down13[0], up21[1], down13[2]])
        up13 = self.up13([up12, down12[0], up22[1], down12[2]])
        up14 = self.up14([up13, down11[0], up23, down11[2]])

        if self.if_ds:
            featureMaps = [up11, up41, up12, up32, up13, up23, torch.cat(up14, dim=1)][::-1]
            return [self.outputs[i](featureMaps[i]) for i in range(7)]
        else:
            return self.outputs[0](torch.cat(up14,dim=1))



if __name__ == '__main__':
    model = MNet(num_classes=4)
    print(model)

