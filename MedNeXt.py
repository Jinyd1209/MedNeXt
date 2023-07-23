"""
@Author: yidong jin
@Email: yidong4242@gmail.com
"""

import torch.nn as nn
import torch


class conv_block(nn.Module):
    def __init__(self, c_in, scale, k_size):
        super().__init__()
        self.dw_conv = nn.Conv3d(c_in, c_in, kernel_size=k_size, groups=c_in, stride=1,
                                 padding=(k_size - 1) // 2, bias=False)
        self.expansion = nn.Conv3d(c_in, c_in * scale, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(c_in * scale, c_in, kernel_size=(1, 1, 1), stride=1)
        self.norm = nn.GroupNorm(32, c_in)

    def forward(self, x):
        identity = x
        out = self.norm(self.dw_conv(x))
        out = self.act(self.expansion(out))
        out = self.compress(out)
        out = torch.add(out, identity)
        return out



class down_block(nn.Module):
    def __init__(self, c_in, scale, k_size):
        super().__init__()
        self.dw_conv = nn.Conv3d(c_in, c_in, kernel_size=k_size, groups=c_in, stride=2, padding=(k_size - 1) // 2,
                                 bias=False)
        self.norm = nn.GroupNorm(32, c_in)
        self.expansion = nn.Conv3d(c_in, scale * c_in, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(scale * c_in, 2 * c_in, kernel_size=(1, 1, 1), stride=1)
        self.shortcut = nn.Conv3d(c_in, 2 * c_in, kernel_size=(1, 1, 1), stride=2)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.norm(self.dw_conv(x))
        out = self.act(self.expansion(out))
        out = self.compress(out)
        out = torch.add(out, shortcut)
        return out



class up_block(nn.Module):
    def __init__(self, c_in, scale, k_size):
        super().__init__()
        self.dw_conv = nn.ConvTranspose3d(c_in, c_in, kernel_size=k_size, stride=2, padding=(k_size - 1) // 2,
                                          output_padding=1, groups=c_in, bias=False)
        self.norm = nn.GroupNorm(32, c_in)
        self.expansion = nn.Conv3d(c_in, c_in * scale, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(c_in * scale, c_in // 2, kernel_size=(1, 1, 1), stride=1)
        self.shortcut = nn.ConvTranspose3d(c_in, c_in // 2, kernel_size=(1, 1, 1), output_padding=1, stride=2)

    def forward(self, x1, x2):
        short = self.shortcut(x1)
        x1 = self.norm(self.dw_conv(x1))
        x1 = self.act(self.expansion(x1))
        x1 = self.compress(x1)
        x1 = torch.add(x1, short)
        out = torch.add(x1, x2)
        return out




class MedNeXt(nn.Module):
    def __init__(self, in_channel, base_c, k_size, num_block, scale, num_class):
        super().__init__()
        self.stem = nn.Conv3d(in_channel, base_c, kernel_size=(1, 1, 1), stride=1)
        self.layer1 = self._make_layer(base_c, num_block[0], scale[0], k_size)
        self.down1 = down_block(base_c, scale[1], k_size)
        self.layer2 = self._make_layer(base_c * 2, num_block[1], scale[1], k_size)
        self.down2 = down_block(base_c * 2, scale[2], k_size)
        self.layer3 = self._make_layer(base_c * 4, num_block[2], scale[2], k_size)
        self.down3 = down_block(base_c * 4, scale[3], k_size)
        self.layer4 = self._make_layer(base_c * 8, num_block[3], scale[3], k_size)
        self.down4 = down_block(base_c * 8, scale[4], k_size)
        self.layer5 = self._make_layer(base_c * 16, num_block[4], scale[4], k_size)

        self.up1 = up_block(base_c * 16, scale[4], k_size)
        self.layer6 = self._make_layer(base_c * 8, num_block[5], scale[5], k_size)
        self.up2 = up_block(base_c * 8, scale[5], k_size)
        self.layer7 = self._make_layer(base_c * 4, num_block[6], scale[6], k_size)
        self.up3 = up_block(base_c * 4, scale[6], k_size)
        self.layer8 = self._make_layer(base_c * 2, num_block[7], scale[7], k_size)
        self.up4 = up_block(base_c * 2, scale[7], k_size)
        self.layer9 = self._make_layer(base_c, num_block[8], scale[8], k_size)

        #deep supervision
        # self.ds4 = nn.Conv3d(base_c*16,num_class,kernel_size=(1,1,1),stride=1)
        # self.ds3 = nn.Conv3d(base_c*8,num_class,kernel_size=(1,1,1),stride=1)
        # self.ds2 = nn.Conv3d(base_c*4,num_class,kernel_size=(1,1,1),stride=1)
        # self.ds1 = nn.Conv3d(base_c*2,num_class,kernel_size=(1,1,1),stride=1)

        self.out = nn.Conv3d(base_c, num_class, kernel_size=(1, 1, 1), stride=1)

    def _make_layer(self, c_in, n_conv, ratio, k_size):
        layers = []
        for _ in range(n_conv):
            layers.append(conv_block(c_in, ratio, k_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        out1 = self.layer1(x)
        d1 = self.down1(out1)
        out2 = self.layer2(d1)
        d2 = self.down2(out2)
        out3 = self.layer3(d2)
        d3 = self.down3(out3)
        out4 = self.layer4(d3)
        d4 = self.down4(out4)
        out_5 = self.layer5(d4)
        # out_ds4 = self.ds4(out_5)
        up1 = self.up1(out_5, out4)
        out6 = self.layer6(up1)
        # out_ds3 = self.ds3(out6)
        up2 = self.up2(out6, out3)
        out7 = self.layer7(up2)
        # out_ds2 = self.ds2(out7)
        up3 = self.up3(out7, out2)
        out8 = self.layer8(up3)
        # out_ds1 = self.ds1(out8)
        up4 = self.up4(out8, out1)
        out9 = self.layer9(up4)
        out = self.out(out9)
        # return out_ds4,out_ds3,out_ds2,out_ds1,out
        return out


def get_mednet():
    num_block = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    scale = [2, 3, 4, 4, 4, 4, 4, 3, 2]
    net = MedNeXt(2, 32, 5, num_block, scale, 2)
    return net




# debug
#if __name__ == '__main__':
    # net = get_mednet()
    #     x = torch.rand(1,2,80,96,80)
    #     out = net(x)
    #     print(out.shape)
