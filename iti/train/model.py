from enum import Enum

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import LayerNorm, init


class DiscriminatorMode(Enum):
    SINGLE = "SINGLE"  # use a single discriminator across all channels
    CHANNELS = "CHANNELS"  # use a discriminator per channel and one for the the combined channels
    SINGLE_PER_CHANNEL = "SINGLE_PER_CHANNEL"  # use a single discriminator for each channel and one for the combined channels


########################## Generator ##################################
class GeneratorAB(nn.Module):
    def __init__(self, input_dim, output_dim, depth, n_upsample, dim=64, output_activ='tanh', skip_connections=True, **kwargs):
        super().__init__()
        self.depth = depth
        # self.skip_connections = skip_connections
        self.from_image = Conv2dBlock(input_dim, dim, 7, 1, 3, **kwargs)
        ##################### Encoder #####################
        self.down_blocks = []
        n_convs = 1
        for i in range(depth):
            self.down_blocks += [DownBlock(dim, 2 * dim, n_convs, **kwargs)]
            dim *= 2
            n_convs = n_convs + 1 if n_convs < 3 else 3
        ##################### Core #####################
        self.core_block = CoreBlock(dim, dim, 3, **kwargs)
        ##################### Decoder #####################
        self.up_blocks = []
        for i in range(depth):
            self.up_blocks += [UpBlock(dim, dim // 2, n_convs, skip_connection=skip_connections, **kwargs)]
            dim //= 2
            n_convs = n_convs -1 if n_convs > 1 else 1
        ##################### Upsampling #####################
        self.sampling_blocks = []
        for _ in range(n_upsample):
            self.sampling_blocks += [UpBlock(dim, dim // 2, 1, skip_connection=False, **kwargs)]
            dim //= 2
        self.to_image = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=output_activ, pad_type=kwargs['pad_type'])

        self.model = nn.Sequential(self.from_image, *self.down_blocks, self.core_block, *self.up_blocks, *self.sampling_blocks, self.to_image)

    def forward(self, x):
        self.skip_connections = True
        x = self.from_image(x)
        # encode
        skip_connections = []
        for down in self.down_blocks:
            x, skip = down(x)
            skip_connections.append(skip)
        # core
        x = self.core_block(x)
        # decode
        for up, skip in zip(self.up_blocks, reversed(skip_connections)):
            if self.skip_connections:
                x = up(x, skip)
            else:
                x = up(x)
        # upsampling
        for up in self.sampling_blocks:
            x = up(x)
        x = self.to_image(x)
        return x

class GeneratorBA(nn.Module):
    def __init__(self, input_dim, output_dim, noise_dim, depth, depth_noise, n_downsample, dim=64, output_activ='tanh', skip_connections=True, **kwargs):
        super().__init__()
        self.depth = depth
        self.noise_dim = noise_dim
        self.depth_noise = depth_noise
        self.n_downsample = n_downsample
        self.skip_connections = skip_connections
        dim = dim // (2 ** n_downsample)
        self.from_image = Conv2dBlock(input_dim, dim, 7, 1, 3, **kwargs)
        ##################### Downsampling #####################
        self.sampling_blocks = []
        for _ in range(n_downsample):
            self.sampling_blocks += [DownBlock(dim, dim * 2, 1, **kwargs)]
            dim *= 2
        ##################### Encoder #####################
        self.down_blocks = []
        n_convs = 1
        for i in range(depth):
            self.down_blocks += [DownBlock(dim, 2 * dim, n_convs, **kwargs)]
            dim *= 2
            n_convs = n_convs + 1 if n_convs < 3 else 3
        ##################### Noise #####################
        self.noise_blocks = []
        self.noise_blocks += [Conv2dBlock(noise_dim, dim, 7, 1, 3, **kwargs)]
        for _ in range(depth_noise - depth):
            self.noise_blocks += [UpBlock(dim, dim, 3, skip_connection=False, **kwargs)]
        self.noise_blocks = nn.Sequential(*self.noise_blocks)
        ##################### Core #####################
        self.core_block = CoreBlock(dim * 2, dim, 3, **kwargs)
        ##################### Decoder #####################
        self.up_blocks = []
        for i in range(depth):
            self.up_blocks += [UpBlock(dim, dim // 2, n_convs, skip_connection=skip_connections, **kwargs)]
            dim //= 2
            n_convs = n_convs - 1 if n_convs > 1 else 1
        self.to_image = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=output_activ, pad_type=kwargs['pad_type'])

        self.model = nn.Sequential(self.from_image, *self.sampling_blocks, *self.down_blocks, self.noise_blocks, self.core_block, *self.up_blocks, self.to_image)

    def forward(self, image, noise):
        x = self.from_image(image)
        # downsampling
        for down in self.sampling_blocks:
            x, _ = down(x)
        # encode
        skip_connections = []
        for down in self.down_blocks:
            x, skip = down(x)
            skip_connections.append(skip)
        # noise
        y = self.noise_blocks(noise)
        # core
        x = torch.cat([x, y], dim=1)
        x = self.core_block(x)
        # decode
        for up, skip in zip(self.up_blocks, reversed(skip_connections)):
            if hasattr(self, 'skip_connections') and self.skip_connections: # check for backwards compatibility
                x = up(x, skip)
            else:
                x = up(x)

        x = self.to_image(x)
        return x

    def forwardRandomNoise(self, image):
        n_gen = Variable(torch.rand(image.size(0), self.noise_dim,
                                    image.size(2) // 2 ** (self.depth_noise + self.n_downsample),
                                    image.size(3) // 2 ** (self.depth_noise + self.n_downsample)).cuda())
        return self.forward(image, noise=n_gen)


class Discriminator(nn.Module):
    def __init__(self, input_dim, n_filters, num_scales=3, depth_discriminator=4,
                 discriminator_mode=DiscriminatorMode.SINGLE,
                 norm='in_rs_aff', batch_statistic=False):
        self.pad_type = 'reflect'
        self.activ = 'relu'
        self.batch_statistic = batch_statistic
        self.norm = norm
        self.depth_discriminator = depth_discriminator
        super().__init__()
        self.input_dim = input_dim
        self.num_scales = num_scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.discs = nn.ModuleList()
        self.channel_discs = nn.ModuleDict()
        # create combined discriminators
        for _ in range(num_scales):
            self.discs.append(self._make_net(input_dim, n_filters))
        # create channel discriminators
        if discriminator_mode == DiscriminatorMode.CHANNELS:
            for i in range(input_dim):
                channel_disc = nn.ModuleList()
                for _ in range(num_scales):
                    channel_disc.append(self._make_net(1, n_filters))
                self.channel_discs['%d' % i] = channel_disc
        if discriminator_mode == DiscriminatorMode.SINGLE_PER_CHANNEL:
            channel_disc = nn.ModuleList()
            for _ in range(num_scales):
                channel_disc.append(self._make_net(1, n_filters))
            for i in range(input_dim):
                self.channel_discs['%d' % i] = channel_disc

    def _make_net(self, input_dim, dim=64):
        cnn_x = []
        if self.batch_statistic:
            cnn_x += [BatchStatistic()]
        cnn_x += [Conv2dBlock((input_dim * 3) if self.batch_statistic else input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.depth_discriminator - 1):
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
        # normalize for Discriminators
        return loss / len(outs0)

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            loss += torch.mean((out0 - 1) ** 2)  # LSGAN
        # normalize for Discriminators
        return loss / len(outs0)

    def calc_content_loss(self, input_real, input_fake):
        loss = []
        for i in range(self.num_scales):
            # content loss of combined discriminator
            x = input_real
            y = input_fake
            for j, layer in enumerate(self.discs[i][:-1]):
                x = layer(x)
                y = layer(y)
                if j == 0: # skip first layer (no normalization)
                    continue
                loss.append(torch.mean(torch.abs(x - y), [1, 2, 3]))

            # content loss of channel discriminator
            for j, discs in enumerate(self.channel_discs.values()):
                x = input_real[:, j:j + 1]
                y = input_fake[:, j:j + 1]
                for k, layer in enumerate(discs[i][:-1]):
                    x = layer(x)
                    y = layer(y)
                    if k == 0:  # skip first layer (no normalization)
                        continue
                    loss.append(torch.mean(torch.abs(x - y), [1, 2, 3]))

            input_real = self.downsample(input_real)
            input_fake = self.downsample(input_fake)
        return torch.mean(torch.stack(loss, 1), 1)

    def calc_content_map(self, input_real, input_fake, skip_last=1):
        loss = []
        for i in range(self.num_scales):
            # content loss of combined discriminator
            x = input_real
            y = input_fake
            for j, layer in enumerate(self.discs[i][:-skip_last]):
                up = nn.UpsamplingBilinear2d(scale_factor=2 ** (i + j + 1))
                x = layer(x)
                y = layer(y)
                if j == 0: # skip first layer (no normalization)
                    continue
                diff = torch.abs(x - y)
                diff = torch.mean(diff, 1).unsqueeze(1)
                loss.append(up(diff))

            # content loss of channel discriminator
            for j, discs in enumerate(self.channel_discs.values()):
                up = nn.UpsamplingBilinear2d(scale_factor=2 ** (i + j + 1))
                x = input_real[:, j:j + 1]
                y = input_fake[:, j:j + 1]
                for k, layer in enumerate(discs[i][:-skip_last]):
                    x = layer(x)
                    y = layer(y)
                    if k == 0:  # skip first layer (no normalization)
                        continue
                    diff = torch.abs(x - y)
                    diff = torch.mean(diff, 1).unsqueeze(1)
                    loss.append(up(diff))

            input_real = self.downsample(input_real)
            input_fake = self.downsample(input_fake)
        return torch.mean(torch.cat(loss, 1), 1).unsqueeze(1)


########################## Encoder / Decoder ##########################

class NoiseEstimator(nn.Module):
    def __init__(self, input_dim, dim, noise_dim, depth, **kwargs):
        super().__init__()

        model = []
        model += [Conv2dBlock(input_dim, dim, 4, 2, 1, norm='none', activation=kwargs['activation'], pad_type=kwargs['pad_type'])]
        for i in range(depth - 1):
            model += [Conv2dBlock(dim, dim * 2, 4, 2, 1, **kwargs)]
            dim *= 2
        model += [nn.Conv2d(dim, noise_dim, 1, 1, 0), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class DownBlock(nn.Module):

    def __init__(self, dim, out_dim, n_convs, **kwargs):
        super().__init__()
        assert n_convs >= 1, 'Invalid configuration, requires at least 1 convolution block'
        self.convs = nn.Sequential(*[Conv2dBlock(dim, dim, 3, 1, 1, **kwargs) for _ in range(n_convs)])
        self.down = Conv2dBlock(dim, out_dim, 3, 2, 1, **kwargs)
        self.module_list = nn.ModuleList([self.convs, self.down])

    def forward(self, x):
        x = self.convs(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):

    def __init__(self, in_dim, out_dim, n_convs, skip_connection=True, **kwargs):
        super().__init__()
        assert n_convs >= 1, 'Invalid configuration, requires at least 1 convolution block'
        self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), Conv2dBlock(in_dim, out_dim, 3, 1, 1, **kwargs))
        self.convs = nn.Sequential(
            *[Conv2dBlock(out_dim * 2 if i == 0 and skip_connection else out_dim, out_dim, 3, 1, 1, **kwargs)
              for i in range(n_convs)])
        self.module_list = nn.ModuleList([self.up, self.convs])

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.convs(x)
        return x

class CoreBlock(nn.Module):

    def __init__(self, in_dim, out_dim, n_convs, **kwargs):
        super().__init__()
        assert n_convs > 1, 'Invalid configuration, requires at least 1 convolution block'
        module = [Conv2dBlock(in_dim, out_dim, 3, 1, 1, **kwargs)]
        module += [Conv2dBlock(out_dim, out_dim, 3, 1, 1, **kwargs) for _ in range(n_convs - 1)]
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)


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
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'in_rs':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True, momentum=0.01)
        elif norm == 'in_aff':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=False, affine=True)
        elif norm == 'in_rs_aff':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True, momentum=0.01, affine=True)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
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
        conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias) if not transpose \
            else nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias)
        if norm == 'sn':
            self.conv = SpectralNorm(conv)
        else:
            self.conv = conv

    def init_conv(self, conv):
        init.kaiming_normal(conv.weight)
        if conv.bias is not None:
            conv.bias.data.zero_()

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class BatchStatistic(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat([x,
                          torch.mean(x, [0, 2, 3], keepdim=True) * torch.ones_like(x),
                          torch.std(x, [0, 2, 3], keepdim=True) * torch.ones_like(x)], 1)

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)
