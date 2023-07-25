from torch import nn
import torch

def make_model(args):
    print('DRRN_s')
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
    def __init__(self, num_features,move_p,move_c):
        super(ResidualUnit, self).__init__()
        self.module = nn.Sequential(
            ConvLayer(num_features, num_features),
            ConvLayer(num_features, num_features)
        )
        self.m_p = move_p
        self.m_c = move_c

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
                zeros[:, :, :, move_pixel:] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2, :,
                                            :W - move_pixel]  # right
            elif direction == 'W-':
                zeros[:, :, :, :-move_pixel] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2, :,
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
                zeros[:, :, :-move_pixel, :] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2 + 1,
                                            move_pixel:, :]  # up
            elif direction == 'H-':
                zeros[:, :, move_pixel:, :] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2 + 1,
                                            :H - move_pixel, :]  # down
            elif direction == 'W+':
                zeros[:, :, :, move_pixel:] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2 + 1,
                                            :,
                                            :W - move_pixel]  # right
            elif direction == 'W-':
                zeros[:, :, :, :-move_pixel] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2 + 1,
                                            :,
                                            move_pixel:]  # left
            else:
                raise NotImplementedError("Direction should be 'H+', 'H-', 'W+', 'W-'.")

            return torch.cat(
                (input[:, 0:mid_channel - move_channel // 2], zeros, input[:, mid_channel + move_channel // 2 + 1:]),
                1)

    def shift_bi_features(self, input, move_pixel, move_channel=0, direction='H'):
        H = input.shape[2]
        W = input.shape[3]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2

        zero_left = torch.zeros_like(input[:, :move_channel])
        zero_right = torch.zeros_like(input[:, :move_channel])
        if direction == 'H':
            zero_left[:, :, :-move_pixel, :] = input[:, mid_channel - move_channel:mid_channel, move_pixel:, :]  # up
            zero_right[:, :, move_pixel:, :] = input[:, mid_channel:mid_channel + move_channel, :H - move_pixel, :]  # down

        elif direction == 'W':
            zero_left[:, :, :, :-move_pixel] = input[:, mid_channel - move_channel:mid_channel, :, move_pixel:]  # left
            zero_right[:, :, :, move_pixel:] = input[:, mid_channel:mid_channel + move_channel, :, :W - move_pixel]  # right

        else:
            raise NotImplementedError("Direction should be 'H' or 'W'.")
        return torch.cat(
            (input[:, 0:mid_channel - move_channel], zero_left, zero_right, input[:, mid_channel + move_channel:]), 1)

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

        zero_left1[:, :, :-move_pixel, :] = input[:, mid_channel - move_channel * 2:mid_channel - move_channel, move_pixel:,:]  # up
        zero_left2[:, :, move_pixel:, :] = input[:, mid_channel - move_channel:mid_channel, :H - move_pixel, :]  # down
        zero_left3[:, :, :-move_pixel, move_pixel:] = input[:, mid_channel - move_channel * 3:mid_channel - move_channel*2,move_pixel:, :W - move_pixel]  # 1
        zero_left4[:, :, :-move_pixel, :-move_pixel] = input[:, mid_channel - move_channel * 4:mid_channel - move_channel*3,  move_pixel:, move_pixel:]  # 2
        zero_right1[:, :, :, :-move_pixel] = input[:, mid_channel:mid_channel + move_channel, :, move_pixel:]  # left
        zero_right2[:, :, :, move_pixel:] = input[:, mid_channel + move_channel:mid_channel + move_channel * 2, :,:W - move_pixel]  # right
        zero_right3[:, :, move_pixel:, :-move_pixel] = input[:, mid_channel+ move_channel*2:mid_channel + move_channel*3, :H - move_pixel, move_pixel:]  # left
        zero_right4[:, :, move_pixel:, move_pixel:] = input[:, mid_channel + move_channel*3:mid_channel + move_channel * 4,  :H - move_pixel, :W - move_pixel]  # right

        return torch.cat(
            (input[:, 0:mid_channel - move_channel * 4], zero_left4, zero_left3, zero_left1, zero_left2, zero_right1, zero_right2,zero_right3,zero_right4,
            input[:, mid_channel + move_channel * 4:]),
            1)


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
            zero_left[:, :, move_pixel:, :] = input[:, mid_channel - move_channel:mid_channel, :H - move_pixel, :]  # down
        else:
            raise NotImplementedError("Direction on H should be '+' or '-'.")
        if w == '+':
            zero_right[:, :, :, move_pixel:] = input[:, mid_channel:mid_channel + move_channel, :, :W - move_pixel]  # right
        elif w == '-':
            zero_right[:, :, :, :-move_pixel] = input[:, mid_channel:mid_channel + move_channel, :, move_pixel:]  # left
        else:
            raise NotImplementedError("Direction on W should be '+' or '-'.")

        return torch.cat(
            (input[:, 0:mid_channel - move_channel], zero_left, zero_right, input[:, mid_channel + move_channel:]), 1)

    def shift_bi_diag_features(self, input, move_pixel, move_channel=0, direction='13'):
        H = input.shape[2]
        W = input.shape[3]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2

        zero_left = torch.zeros_like(input[:, :move_channel])
        zero_right = torch.zeros_like(input[:, :move_channel])
        if direction == '13':
            zero_left[:, :, :-move_pixel, move_pixel:] = input[:,
                                                        mid_channel - move_channel:mid_channel,
                                                        move_pixel:, :W - move_pixel]

            zero_right[:, :, move_pixel:, :-move_pixel] = input[:,
                                                        mid_channel:mid_channel + move_channel,
                                                        move_pixel:, move_pixel:]
        if direction == '24':
            zero_left[:, :, :-move_pixel, :-move_pixel] = input[:,
                                                        mid_channel - move_channel:mid_channel,
                                                        move_pixel:, move_pixel:]
            zero_right[:, :, move_pixel:, move_pixel:] = input[:,
                                                        mid_channel:mid_channel + move_channel,
                                                        :H - move_pixel, :W - move_pixel]
        return torch.cat(
            (input[:, 0:mid_channel - move_channel], zero_left, zero_right, input[:, mid_channel + move_channel:]), 1)

    def shift_uni_diag_features(self, input, move_pixel, move_channel=0, direction='4'):
        if move_channel % 2 == 0:
            H = input.shape[2]
            W = input.shape[3]
            channel_size = input.shape[1]
            mid_channel = channel_size // 2

            zeros = torch.zeros_like(input[:, :move_channel])
            if direction == '1':
                zeros[:, :, :-move_pixel, move_pixel:] = input[:,
                                                        mid_channel - move_channel // 2:mid_channel + move_channel // 2,
                                                        move_pixel:, :W - move_pixel]
            elif direction == '2':
                zeros[:, :, :-move_pixel, :-move_pixel] = input[:,
                                                        mid_channel - move_channel // 2:mid_channel + move_channel // 2,
                                                        move_pixel:, move_pixel:]
            elif direction == '3':
                zeros[:, :, move_pixel:, :-move_pixel] = input[:,
                                                        mid_channel - move_channel // 2:mid_channel + move_channel // 2,
                                                        :H - move_pixel, move_pixel:]
            elif direction == '4':
                zeros[:, :, move_pixel:, move_pixel:] = input[:,
                                                        mid_channel - move_channel // 2:mid_channel + move_channel // 2,
                                                        :H - move_pixel, :W - move_pixel]
            else:
                raise NotImplementedError("Direction should be 1,2,3,4.")

            return torch.cat(
                (input[:, 0:mid_channel - move_channel // 2], zeros, input[:, mid_channel + move_channel // 2:]),
                1)


    
    def shift_cross_diag_features(self, input, move_pixel, move_channel=0, direction='34'):
        H = input.shape[2]
        W = input.shape[3]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2

        zero_left = torch.zeros_like(input[:, :move_channel])
        zero_right = torch.zeros_like(input[:, :move_channel])
        if direction == '12':
            zero_left[:, :, :-move_pixel, move_pixel:] = input[:,
                                                        mid_channel - move_channel:mid_channel,
                                                        move_pixel:, :W - move_pixel]

            zero_right[:, :, :-move_pixel, :-move_pixel] = input[:,
                                                        mid_channel:mid_channel + move_channel,
                                                        move_pixel:, move_pixel:]
        if direction == '23':
            zero_left[:, :, :-move_pixel, :-move_pixel] = input[:,
                                                        mid_channel - move_channel:mid_channel,
                                                        move_pixel:, move_pixel:]
            zero_right[:, :, move_pixel:, :-move_pixel] = input[:,
                                                        mid_channel:mid_channel + move_channel,
                                                        :H - move_pixel, move_pixel:]
        if direction == '34':
            zero_left[:, :, move_pixel:, :-move_pixel] = input[:,
                                                        mid_channel - move_channel:mid_channel,
                                                        :H - move_pixel, move_pixel:]
            zero_right[:, :, move_pixel:, move_pixel:] = input[:,
                                                        mid_channel:mid_channel + move_channel,
                                                        :H - move_pixel, :W - move_pixel]

        if direction == '41':
            zero_left[:, :, move_pixel:, move_pixel:] = input[:,
                                                        mid_channel - move_channel:mid_channel,
                                                        :H - move_pixel, :W - move_pixel]
            zero_right[:, :, :-move_pixel, move_pixel:] = input[:,
                                                        mid_channel:mid_channel + move_channel,
                                                        move_pixel:, :W - move_pixel]

        return torch.cat(
            (input[:, 0:mid_channel - move_channel], zero_left, zero_right, input[:, mid_channel + move_channel:]), 1)
    def forward(self, h0, x):
        #x = self.shift_cross_diag_features(x, self.m_p, self.m_c,'41')
        #x = self.shift_bi_diag_features(x, self.m_p, self.m_c,'24')
        #x = self.shift_all_features(x, self.m_p, self.m_c)
        #x = self.conv_s(x)
        #x = self.shift_cross_features(x, self.m_p, self.m_c,w='-', h='+')
        x = self.shift_bi_features(x, self.m_p, self.m_c,'H')
        #x = self.shift_uni_features(x, self.m_p, self.m_c,'H-')
        #x = self.shift_uni_diag_features(x, self.m_p, self.m_c,'4')
  
        return h0 + self.module(x)


class RecursiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, U,mp,mc):
        super(RecursiveBlock, self).__init__()
        self.U = U
        self.h0 = ConvLayer(in_channels, out_channels)
        self.ru = ResidualUnit(out_channels,mp,mc)

    def forward(self, x):
        h0 = self.h0(x)
        x = h0
        for i in range(self.U):
            x = self.ru(h0, x)
        return x


class DRRN(nn.Module):
    def __init__(self, args,B=1, U=9, num_channels=3, num_features=128):
        super(DRRN, self).__init__()
        self.rbs = nn.Sequential(*[RecursiveBlock(num_channels if i == 0 else num_features, num_features, U,args.move_pixel,args.move_channel) for i in range(B)])
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
        x = nn.functional.interpolate(x,scale_factor=self.scale,mode='bicubic',align_corners=False)
        residual = x
        x = self.rbs(x)
        x = self.rec(x)
        x += residual
        return x