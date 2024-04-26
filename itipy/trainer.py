import logging
import os
from datetime import datetime
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import InstanceNorm2d
from torch.utils.data import DataLoader

from itipy.train.model import GeneratorAB, GeneratorBA, Discriminator, NoiseEstimator, DiscriminatorMode


class Trainer(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, upsampling=0, noise_dim=16, n_filters=64,
                 activation='tanh', norm='in_rs_aff', use_batch_statistic=False,
                 n_discriminators=3, discriminator_mode=DiscriminatorMode.SINGLE,
                 depth_generator=3, depth_discriminator=4, depth_noise=4, skip_connections=True,
                 lambda_discriminator=1, lambda_reconstruction=1, lambda_reconstruction_id=.1,
                 lambda_content=10, lambda_content_id=1, lambda_diversity=1, lambda_noise=1,
                 learning_rate=1e-4):
        super().__init__()

        logging.info("######################### Model Configuration ##########################")
        logging.info("Channels (A/B): %s/%s" % (str(input_dim_a), str(input_dim_b)))
        logging.info("Upsampling: %d" % upsampling)
        logging.info("Generator Depth:   %d" % depth_generator)
        logging.info("Discriminator Depth:   %d" % depth_discriminator)
        logging.info("Noise Depth:   %d" % depth_noise)
        logging.info("Number of Discriminators:   %d" % n_discriminators)
        logging.info("Discriminator Mode:   %s" % discriminator_mode)
        logging.info("Base Number of Filters:   %d" % n_filters)
        logging.info("Activation:   %s" % str(activation))
        logging.info("Normalization:   %s" % str(norm))
        logging.info("Skip Connections:   %s" % str(skip_connections))
        logging.info("Batch Statistic:   %s" % str(use_batch_statistic))
        logging.info("Learning Rate:   %f" % learning_rate)
        logging.info("Lambda Discriminator Loss:   %f" % lambda_discriminator)
        logging.info("Lambda Reconstruction Loss:   %f" % lambda_reconstruction)
        logging.info("Lambda Content Loss:   %f" % lambda_content)
        logging.info("Lambda Reconstruction ID Loss:   %f" % lambda_reconstruction_id)
        logging.info("Lambda Content ID Loss:   %f" % lambda_content_id)
        logging.info("Lambda Noise Loss:   %f" % lambda_noise)
        logging.info("Lambda Diversity Loss:   %f" % lambda_diversity)
        logging.info("START TIME:   %s" % datetime.now())
        logging.info("########################################################################")

        self.noise_dim = noise_dim
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        ############################## MODEL CONFIG ###############################
        self.n_filters = n_filters
        self.depth_generator = depth_generator
        self.depth_discriminator = depth_discriminator
        self.depth_noise = depth_noise
        self.upsampling = upsampling

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
                                  norm=norm, output_activ=activation, pad_type='reflect', skip_connections=skip_connections)  # generator for domain a-->b
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
        self.module_list = nn.ModuleList([self.gen_ab, self.gen_ba, self.dis_a, self.dis_b, self.estimator_noise])

        # Setup the optimizers
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_ab.parameters()) + list(self.gen_ba.parameters()) + list(
            self.estimator_noise.parameters())
        self.dis_opt = torch.optim.Adam(dis_params, lr=learning_rate, betas=(0.5, 0.9))
        self.gen_opt = torch.optim.Adam(gen_params, lr=learning_rate, betas=(0.5, 0.9))

        # Training utils
        self.gen_stack = []
        loss_keys = [
            'iteration',
            'loss_gen_a_identity',
            'loss_gen_b_identity',
            'loss_gen_a_translate',
            'loss_gen_b_translate',
            'loss_gen_adv_a',
            'loss_gen_adv_b',
            'loss_gen_a_content',
            'loss_gen_b_content',
            'loss_gen_a_identity_content',
            'loss_gen_b_identity_content',
            'loss_gen_aba_noise',
            'loss_gen_a_identity_noise',
            'loss_dis_a',
            'loss_dis_b',
            'loss_gen_diversity']
        self.train_loss = {key: [] for key in loss_keys}
        self.valid_loss = {key: [] for key in loss_keys}

        logging.info("Total Parameters Generator (BA/AB): %.02f/%.02f M" %
                     (sum(p.numel() for p in self.gen_ba.parameters()) / 1e6,
                      sum(p.numel() for p in self.gen_ab.parameters()) / 1e6))
        logging.info("Total Parameters Discriminator (A/B): %.02f/%.02f M" %
                     (sum(p.numel() for p in self.dis_a.parameters()) / 1e6,
                      sum(p.numel() for p in self.dis_b.parameters()) / 1e6))
        logging.info("Total Parameters Noise Estimator: %.02f M" %
                     (sum(p.numel() for p in self.estimator_noise.parameters()) / 1e6))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        n_gen = self.generateNoise(x_b)
        x_ab = self.gen_ab(x_a)
        x_ba = self.gen_ba(x_b, n_gen)
        return x_ab, x_ba

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

    def generator_update(self, x_a, x_b):
        self.gen_opt.zero_grad()

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
            x_a_upsample = F.pad(x_a_upsample, [0, 0, 0, 0, 0, c_diff], mode='constant',
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
        self.loss_gen_a_identity = self.recon_criterion(x_a_identity, x_a)
        self.loss_gen_b_identity = self.recon_criterion(x_b_identity, x_b)
        self.loss_gen_a_translate = self.recon_criterion(x_aba, x_a)
        self.loss_gen_b_translate = self.recon_criterion(x_bab, x_b)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # Content loss
        self.loss_gen_a_content = torch.mean(self.dis_a.calc_content_loss(x_a, x_aba))
        self.loss_gen_b_content = torch.mean(self.dis_b.calc_content_loss(x_b, x_bab))
        self.loss_gen_a_identity_content = torch.mean(self.dis_a.calc_content_loss(x_a, x_a_identity))
        self.loss_gen_b_identity_content = torch.mean(self.dis_b.calc_content_loss(x_b, x_b_identity))
        # Noise loss
        self.loss_gen_ba_noise = self.recon_criterion(n_ba, n_gen)
        self.loss_gen_aba_noise = self.recon_criterion(n_aba, n_a)
        self.loss_gen_a_identity_noise = self.recon_criterion(n_a_identity, n_a)
        # Diversity loss
        if self.lambda_diversity > 0:
            with torch.no_grad():  # no double back-prop
                n_gen_2 = self.generateNoise(x_b)
                x_ba_div = self.gen_ba(x_b, n_gen_2).detach()
            diversity_diff = self.dis_a.calc_content_loss(x_ba, x_ba_div)
            self.loss_gen_diversity = torch.mean(
                - torch.log((diversity_diff + 1e-6) / (torch.mean(torch.abs(n_gen - n_gen_2), [1, 2, 3]) + 1e-6)))
        else:
            self.loss_gen_diversity = torch.tensor(0.0)
        # total loss
        self.loss_gen_total = self.lambda_reconstruction_id * (self.loss_gen_a_identity + self.loss_gen_b_identity) + \
                              self.lambda_reconstruction * (self.loss_gen_a_translate + self.loss_gen_b_translate) + \
                              self.lambda_discriminator * (self.loss_gen_adv_a + self.loss_gen_adv_b) + \
                              self.lambda_content * (self.loss_gen_a_content + self.loss_gen_b_content) + \
                              self.lambda_content_id * (
                                      self.loss_gen_a_identity_content + self.loss_gen_b_identity_content) + \
                              self.lambda_noise * (
                                      self.loss_gen_ba_noise + self.loss_gen_aba_noise + self.loss_gen_a_identity_noise) + \
                              self.lambda_diversity * self.loss_gen_diversity
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def validate(self, x_a, x_b):
        self.eval()
        with torch.no_grad():
            # noise init
            n_a = self.estimator_noise(x_a)
            n_gen = self.generateNoise(x_b)

            # identity
            x_b_identity = self.gen_ab(self.downsample(x_b))
            x_a_identity = self.gen_ba(self.upsample(x_a), n_a)
            n_a_identity = self.estimator_noise(x_a_identity)

            # translate 1
            x_ab = self.gen_ab(x_a)
            x_ba = self.gen_ba(x_b, n_gen)

            # translate 2
            x_aba = self.gen_ba(x_ab, n_a)
            x_bab = self.gen_ab(x_ba)
            # noise 2
            n_aba = self.estimator_noise(x_aba)

            # reconstruction loss
            self.valid_loss_gen_a_identity = self.recon_criterion(x_a_identity, x_a)
            self.valid_loss_gen_b_identity = self.recon_criterion(x_b_identity, x_b)
            self.valid_loss_gen_a_translate = self.recon_criterion(x_aba, x_a)
            self.valid_loss_gen_b_translate = self.recon_criterion(x_bab, x_b)
            # GAN loss
            self.valid_loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
            self.valid_loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
            self.valid_loss_dis_a = self.dis_a.calc_gen_loss(x_a)  # use only real images fro validation
            self.valid_loss_dis_b = self.dis_b.calc_gen_loss(x_b)
            # Content loss
            self.valid_loss_gen_a_content = torch.mean(self.dis_a.calc_content_loss(x_a, x_aba))
            self.valid_loss_gen_b_content = torch.mean(self.dis_b.calc_content_loss(x_b, x_bab))
            self.valid_loss_gen_a_identity_content = torch.mean(self.dis_a.calc_content_loss(x_a, x_a_identity))
            self.valid_loss_gen_b_identity_content = torch.mean(self.dis_b.calc_content_loss(x_b, x_b_identity))
            # Noise loss
            self.valid_loss_gen_aba_noise = self.recon_criterion(n_aba, n_a)
            self.valid_loss_gen_a_identity_noise = self.recon_criterion(n_a_identity, n_a)
        self.train()

    def fill_stack(self, x_a, x_b):
        self.eval()
        with torch.no_grad():
            n_gen = self.generateNoise(x_b)
            x_ab = self.gen_ab(x_a).cpu().detach()
            x_ba = self.gen_ba(x_b, n_gen).cpu().detach()
            self.gen_stack.append((x_ab, x_ba))
        self.train()

    def discriminator_update(self, x_a, x_b):
        self.dis_opt.zero_grad()
        # noise
        n_gen = self.generateNoise(x_b)
        # translate
        with torch.no_grad():
            x_ab = self.gen_ab(x_a).cpu().detach()
            x_ba = self.gen_ba(x_b, n_gen).cpu().detach()
            self.gen_stack.append((x_ab, x_ba))
            x_ab, x_ba = self.gen_stack.pop(0)
            x_ab, x_ba = x_ab.cuda(), x_ba.cuda()

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba, x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab, x_b)
        self.loss_dis_total = self.loss_dis_a + self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def generateNoise(self, x_b):
        n_gen = Variable(torch.rand(x_b.size(0), self.noise_dim,
                                    x_b.size(2) // 2 ** (self.depth_noise + self.upsampling),
                                    x_b.size(3) // 2 ** (self.depth_noise + self.upsampling)).cuda())
        return n_gen

    def resume(self, checkpoint_dir, epoch=None):
        path = os.path.join(checkpoint_dir, 'checkpoint.pt') if epoch is None else os.path.join(checkpoint_dir,
                                                                                                'checkpoint_%d.pt' % epoch)
        if not os.path.exists(path):
            return 0
        state_dict = torch.load(path)
        # Load generators
        self.gen_ab.load_state_dict(state_dict['gen_ab'])
        self.gen_ba.load_state_dict(state_dict['gen_ba'])
        self.estimator_noise.load_state_dict(state_dict['noise_est'])
        # Load discriminators
        self.dis_a.load_state_dict(state_dict['disc_a'])
        self.dis_b.load_state_dict(state_dict['disc_b'])
        # Load optimizers
        self.gen_opt.load_state_dict(state_dict['opt_gen'])
        self.dis_opt.load_state_dict(state_dict['opt_dis'])
        # Load history
        last_iteration = state_dict['iteration']
        self.train_loss = state_dict['train_loss'] if 'train_loss' in state_dict else self.train_loss
        self.valid_loss = state_dict['valid_loss'] if 'valid_loss' in state_dict else self.valid_loss
        print('Resume from iteration %d' % last_iteration)
        return last_iteration

    def save(self, checkpoint_dir, iterations):
        # Save generators, discriminators, and optimizers
        state_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_%05d.pt' % (iterations + 1))
        state = {'iteration': iterations + 1,
                 'gen_ab': self.gen_ab.state_dict(),
                 'gen_ba': self.gen_ba.state_dict(),
                 'noise_est': self.estimator_noise.state_dict(),
                 'disc_a': self.dis_a.state_dict(),
                 'disc_b': self.dis_b.state_dict(),
                 'opt_gen': self.gen_opt.state_dict(),
                 'opt_dis': self.dis_opt.state_dict(),
                 'train_loss': self.train_loss,
                 'valid_loss': self.valid_loss}
        torch.save(state, state_path)
        if (iterations + 1) % 20000 == 0:
            torch.save(state, checkpoint_path)
            torch.save(self.gen_ab, os.path.join(checkpoint_dir, 'generator_AB.pt'))
            torch.save(self.gen_ba, os.path.join(checkpoint_dir, 'generator_BA.pt'))

    def updateMomentum(self, momentum):
        for module in self.modules():
            if isinstance(module, InstanceNorm2d):
                module.momentum = momentum

    def startBasicTraining(self, base_dir, ds_A, ds_B, ds_valid_A=None, ds_valid_B=None,
                           plot_settings_A=None, plot_settings_B=None, additional_callbacks=[],
                           iterations=int(1e8), num_workers=8, validation_history=False, batch_size=1):
        self.cuda()
        start_it = self.resume(base_dir)
        # Init Callbacks
        from itipy.callback import HistoryCallback, ProgressCallback, SaveCallback, PlotBAB, PlotABA, ValidationHistoryCallback
        history_callback = HistoryCallback(self, base_dir)
        progress_callback = ProgressCallback(self)
        save_callback = SaveCallback(self, base_dir)
        callbacks = [history_callback, progress_callback, save_callback]
        if ds_valid_B is not None or ds_valid_A is not None:
            prediction_dir = os.path.join(base_dir, 'prediction')
            os.makedirs(prediction_dir, exist_ok=True)
        if ds_valid_B is not None:
            callbacks += [PlotBAB(ds_valid_B.sample(4), self, prediction_dir, log_iteration=1000,
                                  plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]
        if ds_valid_A is not None:
            callbacks += [PlotABA(ds_valid_A.sample(4), self, prediction_dir, log_iteration=1000,
                                  plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]
        if validation_history:
            assert ds_valid_B is not None and ds_valid_A is not None, 'Validation history requires validation data sets!'
            callbacks += [ValidationHistoryCallback(self, ds_valid_A, ds_valid_B, base_dir, 1000)]
        callbacks += additional_callbacks
        # init data loaders
        B_iterator = loop(DataLoader(ds_B, batch_size=batch_size, shuffle=True, num_workers=num_workers))
        A_iterator = loop(DataLoader(ds_A, batch_size=batch_size, shuffle=True, num_workers=num_workers))
        # start update cycle
        for it in range(start_it, iterations):
            self.train()
            if it > 100000: # fix running stats
                self.gen_ab.eval()
                self.gen_ba.eval()
            x_a, x_b = next(A_iterator), next(B_iterator)
            x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
            self.discriminator_update(x_a, x_b)
            #
            x_a, x_b = next(A_iterator), next(B_iterator)
            x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
            self.generator_update(x_a, x_b)
            torch.cuda.synchronize()
            #
            self.eval()
            with torch.no_grad():
                for callback in callbacks:
                    callback(it)


def convertSet(data_set, store_path):
    loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=16)
    for i, sample in enumerate(loader):
        np.save(os.path.join(store_path, '%d.npy' % i), sample[0])


def loop(iterable):
    it = iterable.__iter__()
    #
    while True:
        try:
            yield it.next()
        except StopIteration:
            it = iterable.__iter__()
            yield it.next()
        except Exception as ex:
            logging.error(str(ex))
            continue


def skip_invalid(iterable):
    it = iterable.__iter__()
    #
    while True:
        try:
            yield next(it)
        except StopIteration as ex:
            return
        except (AssertionError, ValueError, Exception) as ex:
            logging.error(str(ex))
            continue
