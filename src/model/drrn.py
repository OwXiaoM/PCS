from torch import nn


def make_model(args):
    return DRRN(args)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvLayer, self).__init__()
        self.module = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        )

    def forward(self, x):
        return self.module(x)


class ResidualUnit(nn.Module):
    def __init__(self, num_features):
        super(ResidualUnit, self).__init__()
        self.module = nn.Sequential(
            ConvLayer(num_features, num_features),
            ConvLayer(num_features, num_features)
        )

    def forward(self, h0, x):
        return h0 + self.module(x)


class RecursiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, U):
        super(RecursiveBlock, self).__init__()
        self.U = U
        self.h0 = ConvLayer(in_channels, out_channels)
        self.ru = ResidualUnit(out_channels)

    def forward(self, x):
        h0 = self.h0(x)
        x = h0
        for i in range(self.U):
            x = self.ru(h0, x)
        return x


class DRRN(nn.Module):
    def __init__(self, args, B=1, U=9, num_channels=3, num_features=128):
        super(DRRN, self).__init__()
        self.rbs = nn.Sequential(
            *[RecursiveBlock(num_channels if i == 0 else num_features, num_features, U) for i in range(B)])
        self.rec = ConvLayer(num_features, num_channels)
        self._initialize_weights()
        self.scale = int(args.scale[0])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        residual = x
        x = self.rbs(x)
        x = self.rec(x)
        x += residual
        return x
