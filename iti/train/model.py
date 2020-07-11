from enum import Enum

import torch
from torch import nn
from torch.nn import LayerNorm
from torch.nn.utils.spectral_norm import SpectralNorm


class DiscriminatorMode(Enum):
    SINGLE = "MULTI_CHANNEL"  # use a single discriminator across all channels
    PER_CHANNEL = "PER_CHANNEL"  # use a discriminator per channel and one for the the combined channels
    SINGLE_PER_CHANNEL = "SINGLE_PER_CHANNEL"  # use a single discriminator for each channel and one for the combined channels


########################## Generator ##################################
class GeneratorAB(nn.Module):
    def __init__(self, input_dim, output_dim, n_downsample, n_upsample, dim=64, n_res=9,
                 norm='in', activ='relu', pad_type='reflect', output_activ='tanh'):
        super().__init__()
        ##################### Encoder #####################
        self.down_blocks = []
        self.down_blocks += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(n_downsample):
            self.down_blocks += [DownBlock(dim, 2 * dim, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        ##################### Transformer #####################
        self.res_blocks = ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)
        ##################### Decoder #####################
        self.up_blocks = []
        for i in range(n_upsample):
            self.up_blocks += [UpBlock(dim, dim // 2, norm='in', activation=activ, pad_type=pad_type)]
            dim //= 2
        self.up_blocks += [
            Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=output_activ, pad_type=pad_type)]

        self.model = nn.Sequential(*self.down_blocks, self.res_blocks, *self.up_blocks)

    def forward(self, x):
        # encode
        skip_connections = []
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)
        # transform
        x = self.res_blocks(x)
        # decode
        for up in self.up_blocks:
            if len(skip_connections) > 0:
                x += skip_connections.pop(-1)
            x = up(x)
        return x


class GeneratorBA(nn.Module):
    def __init__(self, input_dim, noise_dim, output_dim, n_downsample, n_upsample, dim=64, n_res=9,
                 norm='in', activ='relu', pad_type='reflect', output_activ='tanh'):
        assert n_upsample >= 2, 'Found noise depth %d, but minimum is 2' % n_upsample
        super().__init__()
        i_dim = int(dim / 2 ** n_downsample)
        self.image_blocks = []
        self.image_blocks += [Conv2dBlock(input_dim, i_dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(n_downsample + 2):
            self.image_blocks += [DownBlock(i_dim, 2 * i_dim, norm=norm, activation=activ, pad_type=pad_type)]
            i_dim *= 2

        n_dim = int(dim * 2 ** 2)
        self.noise_blocks = []
        self.noise_blocks += [Conv2dBlock(noise_dim, n_dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        self.noise_blocks += [ResBlocks(n_res // 2, n_dim, norm=norm, activation=activ, pad_type=pad_type)]

        for _ in range(n_upsample - 2):
            self.noise_blocks += [UpBlock(n_dim, n_dim, norm=norm, activation=activ, pad_type=pad_type)]

        m_dim = int(dim * 2 ** 2)
        self.merge_blocks = [Conv2dBlock(i_dim + n_dim, m_dim, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type)]
        self.merge_blocks += [ResBlocks(n_res // 2, m_dim, norm=norm, activation=activ, pad_type=pad_type)]

        self.up_blocks = []
        for i in range(2):
            self.up_blocks += [UpBlock(n_dim, n_dim // 2, norm=norm, activation=activ, pad_type=pad_type)]
            n_dim //= 2

        self.up_blocks += [
            Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=output_activ, pad_type=pad_type)]

        self.image_blocks = nn.Sequential(*self.image_blocks)
        self.noise_blocks = nn.Sequential(*self.noise_blocks)
        self.merge_blocks = nn.Sequential(*self.merge_blocks)
        self.up_blocks = nn.Sequential(*self.up_blocks)
        self.model = nn.ModuleList([self.image_blocks, self.noise_blocks, self.merge_blocks, self.up_blocks])

    def forward(self, images, noise):
        x = images
        skip_connections = []
        for down in self.image_blocks:
            x = down(x)
            skip_connections.append(x)

        y = self.noise_blocks(noise)

        x = torch.cat([x, y], dim=1)
        x = self.merge_blocks(x)

        for up in self.up_blocks:
            x += skip_connections.pop(-1)
            x = up(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_scales=3, discriminator_mode=DiscriminatorMode.SINGLE):
        self.pad_type = 'reflect'
        self.activ = 'relu'
        self.norm = 'in'
        self.n_layer = 4
        super().__init__()
        self.input_dim = input_dim
        self.num_scales = num_scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.discs = nn.ModuleList()
        self.channel_discs = nn.ModuleDict()
        # create combined discriminators
        for _ in range(num_scales):
            self.discs.append(self._make_net(input_dim))
        # create channel discriminators
        if discriminator_mode == DiscriminatorMode.PER_CHANNEL:
            for i in range(input_dim):
                channel_disc = nn.ModuleList()
                for _ in range(num_scales):
                    channel_disc.append(self._make_net(1))
                self.channel_discs['%d' % i] = channel_disc
        if discriminator_mode == DiscriminatorMode.SINGLE_PER_CHANNEL:
            channel_disc = nn.ModuleList()
            for _ in range(num_scales):
                channel_disc.append(self._make_net(1))
            for i in range(input_dim):
                self.channel_discs['%d' % i] = channel_disc

    def _make_net(self, input_dim, dim=64):
        cnn_x = []
        cnn_x += [Conv2dBlock(input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for i in range(self.num_scales):
            outputs.append(self.discs[i](x))
            for j, discs in enumerate(self.channel_discs.values()):
                outputs.append(discs[i](x[:, j:j + 1]))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)  # LSGAN
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            loss += torch.mean((out0 - 1) ** 2)  # LSGAN
        return loss

    def calc_content_loss(self, input_real, input_fake):
        loss = []
        for i in range(self.num_scales):
            # content loss of combined discriminator
            x = input_real
            y = input_fake
            for layer in self.discs[i][:-1]:
                x = layer(x)
                y = layer(y)
                loss.append(torch.mean(torch.abs(x - y), [1, 2, 3]))

            # content loss of channel discriminator
            for j, discs in enumerate(self.channel_discs.values()):
                x = input_real[:, j:j + 1]
                y = input_fake[:, j:j + 1]
                for layer in discs[i][:-1]:
                    x = layer(x)
                    y = layer(y)
                    loss.append(torch.mean(torch.abs(x - y), [1, 2, 3]))

            input_real = self.downsample(input_real)
            input_fake = self.downsample(input_fake)
        return torch.mean(torch.stack(loss, 1), 1)


########################## Encoder / Decoder ##########################

class NoiseEstimator(nn.Module):
    def __init__(self, input_dim, n_downsample, dim, noise_dim, norm='in', activ='relu', pad_type='reflect'):
        super().__init__()
        assert n_downsample >= 2, 'The minimum noise depth is 2 but found %d' % n_downsample
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.Conv2d(dim, noise_dim, 1, 1, 0), nn.Sigmoid()]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class DownBlock(nn.Module):
    def __init__(self, input_dim, dim, norm='in', activation='relu', pad_type='reflect'):
        super().__init__()

        self.conv1 = Conv2dBlock(input_dim, dim, 3, 2, 1, norm, activation, pad_type)
        self.conv2 = Conv2dBlock(dim, dim, 3, 1, 1, norm, 'none', pad_type)
        self.shortcut_pool = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.shortcut_conv = Conv2dBlock(input_dim, dim, 1, 1, 0, norm, 'none', pad_type)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        shortcut = self.shortcut_pool(x)
        shortcut = self.shortcut_conv(shortcut)

        x = self.conv1(x)
        x = self.conv2(x)

        x += shortcut
        x = self.activation(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, input_dim, dim, norm='in', activation='relu', pad_type='reflect'):
        super().__init__()

        self.conv1 = Conv2dBlock(input_dim, input_dim, 3, 1, 1, norm, activation, pad_type)
        self.conv2 = Conv2dBlock(input_dim, dim, 4, 2, 1, norm, 'none', pad_type, transpose=True)
        self.shortcut_up = nn.UpsamplingNearest2d(scale_factor=2)
        self.shortcut_conv = Conv2dBlock(input_dim, dim, 1, 1, 0, norm, 'none', pad_type)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        shortcut = self.shortcut_up(x)
        shortcut = self.shortcut_conv(shortcut)

        x = self.conv1(x)
        x = self.conv2(x)
        x += shortcut
        x = self.activation(x)
        return x


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super().__init__()

        self.conv1 = Conv2dBlock(dim, dim, 3, 1, 1, norm, activation, pad_type)
        self.conv2 = Conv2dBlock(dim, dim, 3, 1, 1, norm, 'none', pad_type)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        x = self.activation(x)
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', transpose=False):
        super().__init__()
        self.use_bias = True
        # initialize padding
        if transpose:
            self.pad = nn.ZeroPad2d(0)
        elif pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias) if not transpose \
                else nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = torch.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
