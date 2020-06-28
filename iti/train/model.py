import torch
from torch import nn
from torch.nn import LayerNorm
from torch.nn.utils.spectral_norm import SpectralNorm


########################## Generator ##################################

class GeneratorAB(nn.Module):
    def __init__(self, input_dim, output_dim, n_downsample, n_upsample):
        super().__init__()

        self.encoder = Encoder(n_downsample, 4, input_dim, 64, 'in', 'relu', pad_type='reflect')
        self.decoder = Decoder(n_upsample, 4, self.encoder.output_dim, output_dim, res_norm='in', activ='relu',
                               pad_type='reflect')

    def forward(self, images):
        # reconstruct an image
        x, skip_connections = self.encoder(images)
        images_recon = self.decoder(x, skip_connections)
        return images_recon


class GeneratorBA(nn.Module):
    def __init__(self, input_dim, output_dim, noise_dim, n_downsample, n_upsample):
        super().__init__()
        self.encoder = Encoder(n_downsample, 4, input_dim, 64, 'in', 'relu', pad_type='reflect')
        self.decoder = Decoder(n_upsample, 4, self.encoder.output_dim, output_dim, res_norm='in',
                               activ='relu', pad_type='reflect')
        self.merger = Conv2dBlock(self.encoder.output_dim + noise_dim, self.encoder.output_dim, 1, 1, 0,
                                  activation='relu')

    def forward(self, images, noise):
        # reconstruct an image
        x, skip_connections = self.encoder(images)
        x = torch.cat((x, noise), 1)
        x = self.merger(x)
        images_recon = self.decoder(x, skip_connections)
        return images_recon


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_scales=3):
        self.pad_type = 'reflect'
        self.activ = 'relu'
        self.norm = 'in'
        self.n_layer = 4
        super().__init__()
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self, dim=64):
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
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
        loss = 0
        for model in self.cnns:
            x = input_real
            y = input_fake
            for layer in model[:-1]:
                x = layer(x)
                y = layer(y)
                loss += torch.mean(torch.abs(x - y))
            input_real = self.downsample(input_real)
            input_fake = self.downsample(input_fake)
        return loss


########################## Encoder / Decoder ##########################

class NoiseEstimator(nn.Module):
    def __init__(self, input_dim, n_downsample, dim, noise_dim, norm='in', activ='relu', pad_type='reflect'):
        super().__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.Conv2d(dim, noise_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super().__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        skip_connections = []
        for down in self.model[:-1]:
            x = down(x)
            skip_connections.append(x)
        x = self.model[-1](x)
        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='in', activ='relu', pad_type='zero'):
        super().__init__()
        model = []
        model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            model += [
                Conv2dBlock(dim, dim // 2, 4, 2, 1, norm='in', activation=activ, pad_type=pad_type, transpose=True)]
            dim //= 2
        model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x, skip_connections):
        x = self.model[0](x)
        for up in self.model[1:]:
            if len(skip_connections) > 0:
                skip = skip_connections.pop(-1)
                x += skip
            x = up(x)
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

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


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
