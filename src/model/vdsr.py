"""
import torch
import torch.nn as nn
from model import common

def make_model(args, parent=False):
    num_params = sum(param.numel() for param in (VDSR(args).parameters()))
    print(num_params)
    return VDSR(args)

class VDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(VDSR, self).__init__()
    
        self.scale = int(args.scale[0])
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        #self.url = url['r{}f{}'.format(n_resblocks, n_feats)]
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        def basic_block(in_channels, out_channels, act):
            return common.BasicBlock(
                conv, in_channels, out_channels, kernel_size,
                bias=True, bn=False, act=act
            )

        # define body module
        m_body = []
        m_body.append(basic_block(args.n_colors, n_feats, nn.ReLU(True)))
        for _ in range(n_resblocks - 2):
            m_body.append(basic_block(n_feats, n_feats, nn.ReLU(True)))
        m_body.append(basic_block(n_feats, args.n_colors, None))

        self.body = nn.Sequential(*m_body)

    def forward(self, x):

        x = nn.functional.interpolate(x,scale_factor=self.scale,mode='bicubic',align_corners=False)

        #x = self.sub_mean(x)

        res = self.body(x)
        res += x
        #x = self.add_mean(res)

        return res 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
"""
"""
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


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


class VDSR(nn.Module):
    def __init__(self, n_resblocks, n_feats, scale):
        super(VDSR, self).__init__()
        self.scale = scale

        kernel_size = 3 
        n_colors = 3
        rgb_range = 1
        conv=default_conv

        self.scale = scale

        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        m_body = []
        m_body.append(conv(n_colors, n_feats, kernel_size))
        m_body.append(nn.ReLU(True))
        for _ in range(n_resblocks - 2):
            m_body.append(conv(n_feats, n_feats, kernel_size))
            m_body.append(nn.ReLU(True))
        m_body.append(conv(n_feats, n_colors, kernel_size))

        self.body = nn.Sequential(*m_body)


    def forward(self, x, scale=None):
        if scale is not None and scale != self.scale:
            raise ValueError(f"Network scale is {self.scale}, not {scale}")
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x = self.sub_mean(x)
        res = self.body(x)
        res += x
        x = self.add_mean(res)
        return x 


def vdsr_r20f64(scale, pretrained=False):
    return VDSR(20, 64, scale, pretrained)

def make_model(args):
    return VDSR(20, 64, int(args.scale[0]))
"""

def make_model(args, parent=False):
    num_params = sum(param.numel() for param in (VDSR(args).parameters()))
    print(num_params)
    return VDSR(args)

import torch
import torch.nn as nn
from math import sqrt


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self,args):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.scale = int(args.scale[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = nn.functional.interpolate(x,scale_factor=self.scale,mode='bicubic',align_corners=False)
        residual = x1
        out = self.relu(self.input(x1))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out
    
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                            'whose dimensions in the model are {} and '
                                            'whose dimensions in the checkpoint are {}.'
                                            .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                    .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
