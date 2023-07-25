import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class LF_ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(1, 0.25), res_scale=1):
        super(LF_ResBlock, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv4 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.relu4 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale3 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale4 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)

    def forward(self, x):
        yn = x
        G_yn = self.relu1(x)
        G_yn = self.conv1(G_yn)
        yn_1 = G_yn * self.scale1
        Gyn_1 = self.relu2(yn_1)
        Gyn_1 = self.conv2(Gyn_1)
        yn_2 = Gyn_1 * self.scale2
        yn_2 = yn_2 + yn
        Gyn_2 = self.relu3(yn_2)
        Gyn_2 = self.conv3(Gyn_2)
        yn_3 = Gyn_2 * self.scale3
        yn_3 = yn_3 + yn_1
        Gyn_3 = self.relu4(yn_3)
        Gyn_3 = self.conv4(Gyn_3)
        yn_4 = Gyn_3 * self.scale4
        out = yn_4 + yn_2
        return out


class LF_S_ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(1, 0.25), res_scale=1):
        super(LF_S_ResBlock, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv4 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.relu4 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale3 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale4 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.shit_conv = ShiftConv2d_4(inp_channels=n_feats, move_channels=4, move_pixels=2)

    def forward(self, x):
        yn = x
        x_s = self.shit_conv(x)
        G_yn = self.relu1(x_s)
        G_yn = self.conv1(G_yn)
        yn_1 = G_yn * self.scale1
        Gyn_1 = self.relu2(yn_1)
        Gyn_1 = self.conv2(Gyn_1)
        yn_2 = Gyn_1 * self.scale2
        yn_2 = yn_2 + yn
        Gyn_2 = self.relu3(yn_2)
        Gyn_2 = self.conv3(Gyn_2)
        yn_3 = Gyn_2 * self.scale3
        yn_3 = yn_3 + yn_1
        Gyn_3 = self.relu4(yn_3)
        Gyn_3 = self.conv4(Gyn_3)
        yn_4 = Gyn_3 * self.scale4
        out = yn_4 + yn_2
        return out


class RK_ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(1, 0.25), res_scale=1):
        super(RK_ResBlock, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale3 = nn.Parameter(torch.FloatTensor([-1.0]), requires_grad=True)
        self.scale4 = nn.Parameter(torch.FloatTensor([4.0]), requires_grad=True)
        self.scale5 = nn.Parameter(torch.FloatTensor([1 / 6]), requires_grad=True)

    def forward(self, x):
        yn = x
        k1 = self.relu1(x)
        k1 = self.conv1(k1)
        yn_1 = k1 * self.scale1 + yn
        k2 = self.relu2(yn_1)
        k2 = self.conv2(k2)
        yn_2 = yn + self.scale2 * k2
        yn_2 = yn_2 + k1 * self.scale3
        k3 = self.relu3(yn_2)
        k3 = self.conv3(k3)
        yn_3 = k3 + k2 * self.scale4 + k1
        yn_3 = yn_3 * self.scale5
        out = yn_3 + yn
        return out


class RK_S_ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(1, 0.25), res_scale=1, m_p=2, m_c=8):
        super(RK_S_ResBlock, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale3 = nn.Parameter(torch.FloatTensor([-1.0]), requires_grad=True)
        self.scale4 = nn.Parameter(torch.FloatTensor([4.0]), requires_grad=True)
        self.scale5 = nn.Parameter(torch.FloatTensor([1 / 6]), requires_grad=True)
        self.shit_conv = ShiftConv2d1(inp_channels=n_feats, move_channels=m_c, move_pixels=m_p)

    def forward(self, x):
        yn = x
        x_s = self.shit_conv(x)
        k1 = self.relu1(x_s)
        k1 = self.conv1(k1)
        yn_1 = k1 * self.scale1 + yn
        k2 = self.relu2(yn_1)
        k2 = self.conv2(k2)
        yn_2 = yn + self.scale2 * k2
        yn_2 = yn_2 + k1 * self.scale3
        k3 = self.relu3(yn_2)
        k3 = self.conv3(k3)
        yn_3 = k3 + k2 * self.scale4 + k1
        yn_3 = yn_3 * self.scale5
        out = yn_3 + yn
        return out


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class ResBlock_shift(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, move_pixel=0, move_channel=0):

        super(ResBlock_shift, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.m_p = move_pixel
        self.m_c = move_channel

    def shift_uni_features(self, input, move_pixel, move_channel=0, direction='H+'):

        if move_channel % 2 == 0:
            H = input.shape[2]
            W = input.shape[3]
            channel_size = input.shape[1]
            mid_channel = channel_size // 2

            zeros = torch.zeros_like(input[:, :move_channel])
            if direction == 'H+':
                zeros[:, :, :-move_pixel, :] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2,
                                               move_pixel:, :]  # up
            elif direction == 'H-':
                zeros[:, :, move_pixel:, :] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2,
                                              :H - move_pixel, :]  # down
            elif direction == 'W+':
                zeros[:, :, :, move_pixel:] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2,
                                              :,
                                              :W - move_pixel]  # right
            elif direction == 'W-':
                zeros[:, :, :, :-move_pixel] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2,
                                               :,
                                               move_pixel:]  # left
            else:
                raise NotImplementedError("Direction should be 'H+', 'H-', 'W+', 'W-'.")

            return torch.cat(
                (input[:, 0:mid_channel - move_channel // 2], zeros, input[:, mid_channel + move_channel // 2:]),
                1)

        elif move_channel % 2 != 0:
            H = input.shape[2]
            W = input.shape[3]
            channel_size = input.shape[1]
            mid_channel = channel_size // 2

            zeros = torch.zeros_like(input[:, :move_channel])
            if direction == 'H+':
                zeros[:, :, :-move_pixel, :] = input[:,
                                               mid_channel - move_channel // 2:mid_channel + move_channel // 2 + 1,
                                               move_pixel:, :]  # up
            elif direction == 'H-':
                zeros[:, :, move_pixel:, :] = input[:,
                                              mid_channel - move_channel // 2:mid_channel + move_channel // 2 + 1,
                                              :H - move_pixel, :]  # down
            elif direction == 'W+':
                zeros[:, :, :, move_pixel:] = input[:,
                                              mid_channel - move_channel // 2:mid_channel + move_channel // 2 + 1,
                                              :,
                                              :W - move_pixel]  # right
            elif direction == 'W-':
                zeros[:, :, :, :-move_pixel] = input[:,
                                               mid_channel - move_channel // 2:mid_channel + move_channel // 2 + 1,
                                               :,
                                               move_pixel:]  # left
            else:
                raise NotImplementedError("Direction should be 'H+', 'H-', 'W+', 'W-'.")

            return torch.cat(
                (input[:, 0:mid_channel - move_channel // 2], zeros, input[:, mid_channel + move_channel // 2 + 1:]),
                1)

    def forward(self, x):
        x1 = self.shift_uni_features(x, self.m_p, self.m_c, 'H+')
        res = self.body(x1).mul(self.res_scale)
        res += x

        return res


class ResBlock_shift_bi(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, move_pixel=0, move_channel=0):

        super(ResBlock_shift_bi, self).__init__()

        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.m_p = move_pixel
        self.m_c = move_channel

    def shift_bi_features(self, input, move_pixel, move_channel=0, direction='H'):
        H = input.shape[2]
        W = input.shape[3]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2

        zero_left = torch.zeros_like(input[:, :move_channel])
        zero_right = torch.zeros_like(input[:, :move_channel])
        if direction == 'H':
            zero_left[:, :, :-move_pixel, :] = input[:, mid_channel - move_channel:mid_channel, move_pixel:, :]  # up
            zero_right[:, :, move_pixel:, :] = input[:, mid_channel:mid_channel + move_channel, :H - move_pixel,
                                               :]  # down

        elif direction == 'W':
            zero_left[:, :, :, :-move_pixel] = input[:, mid_channel - move_channel:mid_channel, :, move_pixel:]  # left
            zero_right[:, :, :, move_pixel:] = input[:, mid_channel:mid_channel + move_channel, :,
                                               :W - move_pixel]  # right

        else:
            raise NotImplementedError("Direction should be 'H' or 'W'.")
        return torch.cat(
            (input[:, 0:mid_channel - move_channel], zero_left, zero_right, input[:, mid_channel + move_channel:]), 1)

    def forward(self, x):
        x1 = self.shift_bi_features(x, self.m_p, self.m_c, 'H')
        res = self.body(x1).mul(self.res_scale)
        res += x
        return res


class ResBlock_shift_cross(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, move_pixel=0, move_channel=0):

        super(ResBlock_shift_cross, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.m_p = move_pixel
        self.m_c = move_channel
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def shift_cross_features(self, input, move_pixel=0, move_channel=0, w='+', h='+'):
        H = input.shape[2]
        W = input.shape[3]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2

        zero_left = torch.zeros_like(input[:, :move_channel])
        zero_right = torch.zeros_like(input[:, :move_channel])
        if h == '+':
            zero_left[:, :, :-move_pixel, :] = input[:, mid_channel - move_channel:mid_channel, move_pixel:, :]  # up
        elif h == '-':
            zero_left[:, :, move_pixel:, :] = input[:, mid_channel - move_channel:mid_channel, :H - move_pixel,
                                              :]  # down
        else:
            raise NotImplementedError("Direction on H should be '+' or '-'.")
        if w == '+':
            zero_right[:, :, :, move_pixel:] = input[:, mid_channel:mid_channel + move_channel, :,
                                               :W - move_pixel]  # right
        elif w == '-':
            zero_right[:, :, :, :-move_pixel] = input[:, mid_channel:mid_channel + move_channel, :, move_pixel:]  # left
        else:
            raise NotImplementedError("Direction on W should be '+' or '-'.")

        return torch.cat(
            (input[:, 0:mid_channel - move_channel], zero_left, zero_right, input[:, mid_channel + move_channel:]), 1)

    def forward(self, x):
        x1 = self.shift_cross_features(x, self.m_p, self.m_c, w='-', h='+')
        res = self.body(x1).mul(self.res_scale)
        res += x

        return res


class ResBlock_shift_quad(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, move_pixel=0, move_channel=0):

        super(ResBlock_shift_quad, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.m_p = move_pixel
        self.m_c = move_channel
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def shift_all_features(self, input, move_pixel, move_channel=0):
        H = input.shape[2]
        W = input.shape[3]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2
        zero_left1 = torch.zeros_like(input[:, :move_channel])
        zero_right1 = torch.zeros_like(input[:, :move_channel])
        zero_left2 = torch.zeros_like(input[:, :move_channel])
        zero_right2 = torch.zeros_like(input[:, :move_channel])

        zero_left3 = torch.zeros_like(input[:, :move_channel])
        zero_right3 = torch.zeros_like(input[:, :move_channel])
        zero_left4 = torch.zeros_like(input[:, :move_channel])
        zero_right4 = torch.zeros_like(input[:, :move_channel])

        zero_left1[:, :, :-move_pixel, :] = input[:, mid_channel - move_channel * 2:mid_channel - move_channel,
                                            move_pixel:, :]  # up
        zero_left2[:, :, move_pixel:, :] = input[:, mid_channel - move_channel:mid_channel, :H - move_pixel, :]  # down
        zero_left3[:, :, :-move_pixel, move_pixel:] = input[:,
                                                      mid_channel - move_channel * 3:mid_channel - move_channel * 2,
                                                      move_pixel:, :W - move_pixel]  # 1
        zero_left4[:, :, :-move_pixel, :-move_pixel] = input[:,
                                                       mid_channel - move_channel * 4:mid_channel - move_channel * 3,
                                                       move_pixel:, move_pixel:]  # 2
        zero_right1[:, :, :, :-move_pixel] = input[:, mid_channel:mid_channel + move_channel, :, move_pixel:]  # left
        zero_right2[:, :, :, move_pixel:] = input[:, mid_channel + move_channel:mid_channel + move_channel * 2, :,
                                            :W - move_pixel]  # right
        zero_right3[:, :, move_pixel:, :-move_pixel] = input[:,
                                                       mid_channel + move_channel * 2:mid_channel + move_channel * 3,
                                                       :H - move_pixel, move_pixel:]  # left
        zero_right4[:, :, move_pixel:, move_pixel:] = input[:,
                                                      mid_channel + move_channel * 3:mid_channel + move_channel * 4,
                                                      :H - move_pixel, :W - move_pixel]  # right

        return torch.cat(
            (input[:, 0:mid_channel - move_channel * 4], zero_left4, zero_left3, zero_left1, zero_left2, zero_right1,
             zero_right2, zero_right3, zero_right4,
             input[:, mid_channel + move_channel * 4:]),
            1)

    def shift_quad_features(self, input, move, m_c=0):
        H = input.shape[2]
        W = input.shape[3]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2

        zero_left1 = torch.zeros_like(input[:, :m_c])
        zero_right1 = torch.zeros_like(input[:, :m_c])
        zero_left2 = torch.zeros_like(input[:, :m_c])
        zero_right2 = torch.zeros_like(input[:, :m_c])

        zero_left1[:, :, :-move, :] = input[:, mid_channel - m_c * 2:mid_channel - m_c, move:, :]  # up
        zero_left2[:, :, move:, :] = input[:, mid_channel - m_c:mid_channel, :H - move, :]  # down
        zero_right1[:, :, :, :-move] = input[:, mid_channel:mid_channel + m_c, :, move:]  # left
        zero_right2[:, :, :, move:] = input[:, mid_channel + m_c:mid_channel + m_c * 2, :, :W - move]  # right

        return torch.cat(
            (input[:, 0:mid_channel - m_c * 2], zero_left1, zero_left2, zero_right1, zero_right2,
             input[:, mid_channel + m_c * 2:]),
            1)

    def forward(self, x):
        # x1 = self.shift_all_features(x, self.m_p, self.m_c)
        x1 = self.shift_quad_features(x, self.m_p, self.m_c)
        res = self.body(x1).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
