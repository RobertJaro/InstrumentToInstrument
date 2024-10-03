from random import randint
from typing import Dict, Any

import torch
from lightning import LightningModule
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import pad

from itipy.train.model import DiscriminatorMode, GeneratorAB, GeneratorBA, Discriminator, NoiseEstimator


class ITIModule(LightningModule):
    """
    ITI module for the ITI framework. The module contains the following networks:
    - GeneratorAB: Generator for domain A to B
    - GeneratorBA: Generator for domain B to A
    - Discriminator: Discriminator for domain A and B
    - NoiseEstimator: Noise estimator for domain A

    Args:
        input_dim_a (int): Input dimension for domain A.
        input_dim_b (int): Input dimension for domain B.
        upsampling (int): Upsampling factor.
        noise_dim (int): Dimension of the noise.
        n_filters (int): Number of filters.
        activation (str): Activation function.
        norm (str): Normalization.
        use_batch_statistic (bool): Use batch statistic.
        n_discriminators (int): Number of discriminators.
        discriminator_mode (DiscriminatorMode): Discriminator mode.
        depth_generator (int): Depth of the generator.
        depth_discriminator (int): Depth of the discriminator.
        depth_noise (int): Depth of the noise.
        skip_connections (bool): Use skip connections.
        lambda_discriminator (float): Weight for the discriminator loss.
        lambda_reconstruction (float): Weight for the reconstruction loss.
        lambda_reconstruction_id (float): Weight for the reconstruction identity loss.
        lambda_content (float): Weight for the content loss.
        lambda_content_id (float): Weight for the content identity loss.
        lambda_diversity (float): Weight for the diversity loss.
        lambda_noise (float): Weight for the noise loss.
        learning_rate (float): Learning rate.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, input_dim_a=1, input_dim_b=1, upsampling=0, noise_dim=16, n_filters=64,
                 activation='tanh', norm='in_rs_aff', use_batch_statistic=False,
                 n_discriminators=3, discriminator_mode=DiscriminatorMode.SINGLE,
                 depth_generator=3, depth_discriminator=4, depth_noise=4, skip_connections=True,
                 lambda_discriminator=1, lambda_reconstruction=1, lambda_reconstruction_id=.1,
                 lambda_content=10, lambda_content_id=1, lambda_diversity=1, lambda_noise=1,
                 learning_rate=1e-4, **kwargs):
        super().__init__()

        self.noise_dim = noise_dim
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        ############################## MODEL CONFIG ###############################
        self.n_filters = n_filters
        self.depth_generator = depth_generator
        self.depth_discriminator = depth_discriminator
        self.depth_noise = depth_noise
        self.upsampling = upsampling

        ############################## MODEL CONFIG ###############################
        self.learning_rate = learning_rate

        ############################## LOSS WEIGHTS ###############################
        self.lambda_discriminator = lambda_discriminator
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_reconstruction_id = lambda_reconstruction_id
        self.lambda_content_id = lambda_content_id
        self.lambda_content = lambda_content
        self.lambda_diversity = lambda_diversity
        self.lambda_noise = lambda_noise

        ############################## INIT NETWORKS ###############################
        self.gen_ab = GeneratorAB(input_dim_a, input_dim_b, depth_generator, upsampling, n_filters,
                                  norm=norm, output_activ=activation, pad_type='reflect',
                                  skip_connections=skip_connections)  # generator for domain a-->b
        self.gen_ba = GeneratorBA(input_dim_b, input_dim_a, noise_dim, depth_generator, depth_noise, upsampling,
                                  n_filters, norm=norm, output_activ=activation,
                                  pad_type='reflect', skip_connections=skip_connections)  # generator for domain b-->a
        self.dis_a = Discriminator(input_dim_a, n_filters, n_discriminators,
                                   depth_discriminator, discriminator_mode,
                                   norm=norm, batch_statistic=use_batch_statistic)  # discriminator for domain a
        self.dis_b = Discriminator(input_dim_b, n_filters // 2 ** upsampling, n_discriminators,
                                   depth_discriminator + upsampling, discriminator_mode,
                                   norm=norm, batch_statistic=use_batch_statistic)  # discriminator for domain b
        self.estimator_noise = NoiseEstimator(input_dim_a, n_filters, noise_dim,
                                              depth_noise, norm=norm, activation='relu', pad_type='reflect')
        self.downsample = nn.AvgPool2d(2 ** upsampling)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2 ** upsampling)

        # Training utils
        self.gen_stack = []

        self.valid_loss_gen_a_translate = []
        self.valid_loss_gen_b_translate = []
        self.valid_loss_gen_adv_a = []
        self.valid_loss_gen_adv_b = []
        self.valid_loss_dis_a = []
        self.valid_loss_dis_b = []
        # set to manual optimization
        self.automatic_optimization = False

    def configure_optimizers(self):
        # Setup the optimizers
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_ab.parameters()) + list(self.gen_ba.parameters()) + list(
            self.estimator_noise.parameters())
        dis_opt = torch.optim.Adam(dis_params, lr=self.learning_rate, betas=(0.5, 0.9))
        gen_opt = torch.optim.Adam(gen_params, lr=self.learning_rate, betas=(0.5, 0.9))
        return gen_opt, dis_opt

    def training_step(self, batch):
        if self.global_step > 100000:  # fix running stats
            self.gen_ab.eval()
            self.gen_ba.eval()
        x_a, x_b = batch['dis_A'], batch['dis_B']
        if torch.any(torch.std(x_a, dim=(2, 3)) == 0) or torch.any(torch.std(x_b, dim=(2, 3)) == 0):
            print('Skip invalid batch')
        disc_loss_dict = self.discriminator_update(x_a, x_b)
        #
        x_a, x_b = batch['gen_A'], batch['gen_B']
        if torch.any(torch.std(x_a, dim=(2, 3)) == 0) or torch.any(torch.std(x_b, dim=(2, 3)) == 0):
            print('Skip invalid batch')
        train_loss_dict = self.generator_update(x_a, x_b)

        loss_dict = {**disc_loss_dict, **train_loss_dict}
        for k, v in loss_dict.items():
            assert not torch.isnan(v), f'Loss {k} is NaN'

        self.log_dict(loss_dict)


    def validation_step(self, batch, batch_nb, dataloader_idx):
        if dataloader_idx == 0:
            x_a = batch
            n_a = self.estimator_noise(x_a)
            x_ab = self.gen_ab(x_a)
            x_aba = self.gen_ba(x_ab, n_a)
            # reconstruction loss
            valid_loss_gen_a_translate = self.recon_criterion(x_aba, x_a)

            # GAN loss
            valid_loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
            valid_loss_dis_a = self.dis_a.calc_gen_loss(x_a)  # use only real images for validation

            self.valid_loss_gen_a_translate.append(valid_loss_gen_a_translate.detach().cpu())
            self.valid_loss_gen_adv_b.append(valid_loss_gen_adv_b.detach().cpu())
            self.valid_loss_dis_a.append(valid_loss_dis_a.detach().cpu())



        elif dataloader_idx == 1:
            x_b = batch
            n_gen = self.generateNoise(x_b)
            x_ba = self.gen_ba(x_b, n_gen)
            x_bab = self.gen_ab(x_ba)
            valid_loss_gen_b_translate = self.recon_criterion(x_bab, x_b)
            valid_loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
            valid_loss_dis_b = self.dis_b.calc_gen_loss(x_b)

            self.valid_loss_gen_b_translate.append(valid_loss_gen_b_translate.detach().cpu())
            self.valid_loss_gen_adv_a.append(valid_loss_gen_adv_a.detach().cpu())
            self.valid_loss_dis_b.append(valid_loss_dis_b.detach().cpu())

        else:
            raise NotImplementedError('Validation data loader not supported!')

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        valid_loss_gen_a_translate = torch.stack(self.valid_loss_gen_a_translate).mean()
        valid_loss_gen_b_translate = torch.stack(self.valid_loss_gen_b_translate).mean()
        valid_loss_gen_adv_a = torch.stack(self.valid_loss_gen_adv_a).mean()
        valid_loss_gen_adv_b = torch.stack(self.valid_loss_gen_adv_b).mean()
        valid_loss_dis_a = torch.stack(self.valid_loss_dis_a).mean()
        valid_loss_dis_b = torch.stack(self.valid_loss_dis_b).mean()

        self.log_dict({'valid_loss_gen_a_translate': valid_loss_gen_a_translate,
                          'valid_loss_gen_b_translate': valid_loss_gen_b_translate,
                          'valid_loss_gen_adv_a': valid_loss_gen_adv_a,
                          'valid_loss_gen_adv_b': valid_loss_gen_adv_b,
                          'valid_loss_dis_a': valid_loss_dis_a,
                          'valid_loss_dis_b': valid_loss_dis_b,
                          })

        self.valid_loss_gen_a_translate.clear()
        self.valid_loss_gen_b_translate.clear()
        self.valid_loss_gen_adv_a.clear()
        self.valid_loss_gen_adv_b.clear()
        self.valid_loss_dis_a.clear()
        self.valid_loss_dis_b.clear()

        return {'valid_loss_gen_a_translate': valid_loss_gen_a_translate,
                'valid_loss_gen_b_translate': valid_loss_gen_b_translate,
                'valid_loss_gen_adv_a': valid_loss_gen_adv_a,
                'valid_loss_gen_adv_b': valid_loss_gen_adv_b,
                'valid_loss_dis_a': valid_loss_dis_a,
                'valid_loss_dis_b': valid_loss_dis_b,
                }



    def discriminator_update(self, x_a, x_b):
        # noise
        n_gen = self.generateNoise(x_b)
        # translate
        with torch.no_grad():
            x_ab = self.gen_ab(x_a).cpu().detach()
            x_ba = self.gen_ba(x_b, n_gen).cpu().detach()
            self.gen_stack.append((x_ab, x_ba))
            x_ab, x_ba = self.gen_stack.pop(0)
            x_ab, x_ba = x_ab.to(self.device), x_ba.to(self.device)

        # D loss
        loss_dis_a = self.dis_a.calc_dis_loss(x_ba, x_a)
        loss_dis_b = self.dis_b.calc_dis_loss(x_ab, x_b)
        loss_dis_total = loss_dis_a + loss_dis_b

        dis_opt = self.optimizers()[1]
        dis_opt.zero_grad()
        self.manual_backward(loss_dis_total)
        dis_opt.step()

        train_loss_dict = {
            'loss_dis_a': loss_dis_a,
            'loss_dis_b': loss_dis_b,
            'loss_dis_total': loss_dis_total}

        return train_loss_dict

    def generator_update(self, x_a, x_b):
        # noise init
        n_a = self.estimator_noise(x_a)
        n_gen = self.generateNoise(x_b)

        # identity
        if self.input_dim_a > self.input_dim_b:
            x_b_downsample = self.downsample(x_b)
            x_b_downsample = torch.repeat_interleave(x_b_downsample, repeats=self.input_dim_a, dim=1)
            x_b_identity = self.gen_ab(x_b_downsample)

            x_a_upsample = self.upsample(x_a)
            idx = randint(0, self.input_dim_a - 1)
            x_a_upsample = x_a_upsample[:, idx:idx + 1]
            x_a_identity = self.gen_ba(x_a_upsample, n_a)
        else:  # channel difference 0 == no change of dims
            c_diff = self.input_dim_b - self.input_dim_a

            x_b_downsample = self.downsample(x_b)
            x_b_downsample = x_b_downsample[:, :-c_diff] if c_diff > 0 else x_b_downsample
            x_b_identity = self.gen_ab(x_b_downsample)

            x_a_upsample = self.upsample(x_a)
            x_a_upsample = pad(x_a_upsample, [0, 0, 0, 0, 0, c_diff], mode='constant',
                               value=0) if c_diff > 0 else x_a_upsample
            x_a_identity = self.gen_ba(x_a_upsample, n_a)

        n_a_identity = self.estimator_noise(x_a_identity)

        # translate 1
        x_ab = self.gen_ab(x_a)
        x_ba = self.gen_ba(x_b, n_gen)
        # noise 1
        n_ba = self.estimator_noise(x_ba)

        # translate 2
        x_aba = self.gen_ba(x_ab, n_a)
        x_bab = self.gen_ab(x_ba)
        # noise 2
        n_aba = self.estimator_noise(x_aba)

        # reconstruction loss
        loss_gen_a_identity = self.recon_criterion(x_a_identity, x_a)
        loss_gen_b_identity = self.recon_criterion(x_b_identity, x_b)
        loss_gen_a_translate = self.recon_criterion(x_aba, x_a)
        loss_gen_b_translate = self.recon_criterion(x_bab, x_b)
        # GAN loss
        loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # Content loss
        loss_gen_a_content = torch.mean(self.dis_a.calc_content_loss(x_a, x_aba))
        loss_gen_b_content = torch.mean(self.dis_b.calc_content_loss(x_b, x_bab))
        loss_gen_a_identity_content = torch.mean(self.dis_a.calc_content_loss(x_a, x_a_identity))
        loss_gen_b_identity_content = torch.mean(self.dis_b.calc_content_loss(x_b, x_b_identity))
        # Noise loss
        loss_gen_ba_noise = self.recon_criterion(n_ba, n_gen)
        loss_gen_aba_noise = self.recon_criterion(n_aba, n_a)
        loss_gen_a_identity_noise = self.recon_criterion(n_a_identity, n_a)
        # create loss directory
        train_loss_dict = {
            'loss_gen_a_identity': loss_gen_a_identity,
            'loss_gen_b_identity': loss_gen_b_identity,
            'loss_gen_a_translate': loss_gen_a_translate,
            'loss_gen_b_translate': loss_gen_b_translate,
            'loss_gen_adv_a': loss_gen_adv_a,
            'loss_gen_adv_b': loss_gen_adv_b,
            'loss_gen_a_content': loss_gen_a_content,
            'loss_gen_b_content': loss_gen_b_content,
            'loss_gen_a_identity_content': loss_gen_a_identity_content,
            'loss_gen_b_identity_content': loss_gen_b_identity_content,
            'loss_gen_ba_noise': loss_gen_ba_noise,
            'loss_gen_aba_noise': loss_gen_aba_noise,
            'loss_gen_a_identity_noise': loss_gen_a_identity_noise
        }
        # compute total loss
        total_loss = self.lambda_reconstruction_id * (loss_gen_a_identity + loss_gen_b_identity) + \
                     self.lambda_reconstruction * (loss_gen_a_translate + loss_gen_b_translate) + \
                     self.lambda_discriminator * (loss_gen_adv_a + loss_gen_adv_b) + \
                     self.lambda_content * (loss_gen_a_content + loss_gen_b_content) + \
                     self.lambda_content_id * (
                             loss_gen_a_identity_content + loss_gen_b_identity_content) + \
                     self.lambda_noise * (loss_gen_ba_noise + loss_gen_aba_noise + loss_gen_a_identity_noise)
            # Diversity loss
        if self.lambda_diversity > 0:
            with torch.no_grad():  # no double back-prop
                n_gen_2 = self.generateNoise(x_b)
                x_ba_div = self.gen_ba(x_b, n_gen_2).detach()
            diversity_diff = self.dis_a.calc_content_loss(x_ba, x_ba_div)
            loss_gen_diversity = torch.mean(- torch.log((diversity_diff + 1e-6) /
                                                        (torch.mean(torch.abs(n_gen - n_gen_2), [1, 2, 3]) + 1e-6)))
            train_loss_dict['loss_gen_diversity'] = loss_gen_diversity
            total_loss += self.lambda_diversity * loss_gen_diversity

        # update
        gen_opt = self.optimizers()[0]
        gen_opt.zero_grad()
        self.manual_backward(total_loss)
        gen_opt.step()
        return train_loss_dict

    def generateNoise(self, x_b):
        n_gen = Variable(torch.rand(x_b.shape[0], self.noise_dim,
                                    x_b.shape[2] // 2 ** (self.depth_noise + self.upsampling),
                                    x_b.shape[3] // 2 ** (self.depth_noise + self.upsampling)))
        n_gen = n_gen.to(x_b.device)
        return n_gen

    def forwardAB(self, x_a):
        x_ab = self.gen_ab(x_a)
        return x_ab

    def forwardBA(self, x_b):
        n_gen = self.generateNoise(x_b)
        x_ba = self.gen_ba(x_b, n_gen)
        return x_ba

    def forwardABA(self, x_a):
        n_a = self.estimator_noise(x_a)
        x_ab = self.gen_ab(x_a)
        x_aba = self.gen_ba(x_ab, n_a)
        return x_ab, x_aba

    def forwardBAB(self, x_b):
        n_gen = self.generateNoise(x_b)
        x_ba = self.gen_ba(x_b, n_gen)
        x_bab = self.gen_ab(x_ba)
        return x_ba, x_bab

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))
