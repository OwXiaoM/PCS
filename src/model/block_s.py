import torch.nn as nn

import torch
import torch.nn.functional as F


class ShiftConv2d_4(nn.Module):
    def __init__(self, inp_channels, move_channels, move_pixels):
        super(ShiftConv2d_4, self).__init__()
        self.inp_channels = inp_channels
        self.move_p = move_pixels
        self.move_c = move_channels
        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        mid_channel = inp_channels // 2
        up_channels = (mid_channel - move_channels * 2, mid_channel - move_channels)
        down_channels = (mid_channel - move_channels, mid_channel)
        left_channels = (mid_channel, mid_channel + move_channels)
        right_channels = (mid_channel + move_channels, mid_channel + move_channels * 2)
        self.weight[left_channels[0]:left_channels[1], 0, 1, 2] = 1.0  ## left
        self.weight[right_channels[0]:right_channels[1], 0, 1, 0] = 1.0  ## right
        self.weight[up_channels[0]:up_channels[1], 0, 2, 1] = 1.0  ## up
        self.weight[down_channels[0]:down_channels[1], 0, 0, 1] = 1.0  ## down
        self.weight[0:mid_channel - move_channels * 2, 0, 1, 1] = 1.0  ## identity
        self.weight[mid_channel + move_channels * 2:, 0, 1, 1] = 1.0  ## identity
        # print(self.weight)
        # print(self.weight.size())

    def forward(self, x):
        for i in range(self.move_p):
            x = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        return x


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)
        self.sf = ShiftConv2d_4(inp_channels=in_channels, move_channels=2, move_pixels=2)

    def shift_bi_features(self, input, move_pixel=2, move_channel=4, direction='H'):
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

    def shift_quad_features(self, input, move=2, m_c=4):
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

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(self.sf(input)))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(self.sf(r_c1)))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(self.sf(r_c2)))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)
