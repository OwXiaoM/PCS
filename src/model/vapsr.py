
# VAst-receptive-field Pixel attention network
import torch.nn as nn
import torch
import torch.nn.init as init

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn

class VAB(nn.Module):
    def __init__(self, d_model, d_atten):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_atten, 1)
        self.activation = nn.GELU()
        self.atten_branch = Attention(d_atten)
        self.proj_2 = nn.Conv2d(d_atten, d_model, 1)
        self.pixel_norm = nn.LayerNorm(d_model)
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.atten_branch(x)
        x = self.proj_2(x)
        x = x + shorcut

        x = x.permute(0, 2, 3, 1) #(B, H, W, C)
        x = self.pixel_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() #(B, C, H, W)

        return x

def pixelshuffle(in_channels, out_channels, upscale_factor=4):
    upconv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(2)
    upconv2 = nn.Conv2d(16, out_channels * 4, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, pixel_shuffle, lrelu, upconv2, pixel_shuffle])

#both scale X2 and X3 use this version
def pixelshuffle_single(in_channels, out_channels, upscale_factor=2):
    upconv1 = nn.Conv2d(in_channels, 56, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    upconv2 = nn.Conv2d(56, out_channels * upscale_factor * upscale_factor, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, lrelu, upconv2, pixel_shuffle])


def make_layer(block, n_layers, *kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(*kwargs))
    return nn.Sequential(*layers)

class vapsr(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=48, num_block=21, d_atten=64, conv_groups=1):
        super(vapsr, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(VAB, num_block, num_feat, d_atten)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=conv_groups) #conv_groups=2 for VapSR-S

        # upsample
        if scale == 4:
            self.upsampler = pixelshuffle(num_feat, num_out_ch, upscale_factor=scale)
        else:
            self.upsampler = pixelshuffle_single(num_feat, num_out_ch, upscale_factor=scale)

    def forward(self, feat):
        feat = self.conv_first(feat)
        body_feat = self.body(feat)
        body_out = self.conv_body(body_feat)
        feat = feat + body_out
        out = self.upsampler(feat)
        return out

def make_model(args, parent=False):
    model = vapsr(scale=args.scale[0])
    return model