import glob
import logging
import os
from datetime import datetime
from enum import Enum

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from iti.train.model import GeneratorAB, GeneratorBA, Discriminator, NoiseEstimator


class DiscriminatorMode(Enum):
    SINGLE = "MULTI_CHANNEL"  # use a single discriminator across all channels
    PER_CHANNEL = "PER_CHANNEL"  # use a discriminator per channel and one for the the combined channels
    SINGLE_PER_CHANNEL = "SINGLE_PER_CHANNEL"  # use a single discriminator for each channel and one for the combined channels


class Trainer(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, upsampling=0, noise_dim=16, n_filters=64, res_blocks=9,
                 activation='tanh', n_discriminators=3, discriminator_mode=DiscriminatorMode.SINGLE,
                 depth_generator=3, depth_discriminator=4, depth_noise=4,
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
        logging.info("Residual Blocks:   %d" % res_blocks)
        logging.info("Base Number of Filters:   %d" % n_filters)
        logging.info("Activation:   %s" % str(activation))
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
        self.res_blocks = res_blocks
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
        self.gen_ab = GeneratorAB(input_dim_a, input_dim_b, depth_generator,
                                  depth_generator + upsampling, n_filters)  # generator for domain a-->b
        self.gen_ba = GeneratorBA(input_dim_b, noise_dim, input_dim_a, upsampling, depth_noise, n_filters)  # generator for domain b-->a
        self.dis_a = Discriminator(input_dim_a, n_discriminators)  # discriminator for domain a
        self.dis_b = Discriminator(input_dim_b, n_discriminators)  # discriminator for domain b
        self.estimator_noise = NoiseEstimator(input_dim_a, depth_noise, n_filters, noise_dim)
        self.downsample = nn.AvgPool2d(3 ** upsampling, stride=2 ** upsampling, padding=[upsampling, upsampling], count_include_pad=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2 ** upsampling)

        # Setup the optimizers
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_ab.parameters()) + list(self.gen_ba.parameters()) + list(
            self.estimator_noise.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=learning_rate, betas=(0.5, 0.9))
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=learning_rate, betas=(0.5, 0.9))

        # Training utils
        self.gen_stack = []

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        n_gen = self.generateNoise(x_b)
        x_ab = self.gen_ab(x_a)
        x_ba = self.gen_ba(x_b, n_gen)
        self.train()
        return x_ab, x_ba

    def forwardAB(self, x_a):
        self.eval()
        x_ab = self.gen_ab(x_a)
        self.train()
        return x_ab

    def forwardBA(self, x_b):
        self.eval()
        n_gen = self.generateNoise(x_b)
        x_ba = self.gen_ba(x_b, n_gen)
        self.train()
        return x_ba

    def forwardABA(self, x_a):
        self.eval()
        n_a = self.estimator_noise(x_a)
        x_ab = self.gen_ab(x_a)
        x_aba = self.gen_ba(x_ab, n_a)
        self.train()
        return x_ab, x_aba

    def forwardBAB(self, x_b):
        self.eval()
        n_gen = self.generateNoise(x_b)
        x_ba = self.gen_ba(x_b, n_gen)
        x_bab = self.gen_ab(x_ba)
        self.train()
        return x_ba, x_bab

    def generator_update(self, x_a, x_b):
        self.gen_opt.zero_grad()

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
        # noise 1
        n_ba = self.estimator_noise(x_ba)

        # translate 2
        x_aba = self.gen_ba(x_ab, n_a)
        x_bab = self.gen_ab(x_ba)
        # noise 2
        n_aba = self.estimator_noise(x_aba)

        # diversity
        n_gen_2 = self.generateNoise(x_b)
        x_ba_div = self.gen_ba(x_b, n_gen_2)


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
        diversity_diff = self.dis_a.calc_content_loss(x_ba, x_ba_div)
        self.loss_gen_diversity = torch.mean(- torch.log(diversity_diff / (torch.mean(torch.abs(n_gen - n_gen_2), [1, 2, 3]))))
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

    def fill_stack(self, inputs):
        for x_a, x_b in inputs:
            with torch.no_grad():
                n_gen = self.generateNoise(x_b)
                x_ab = self.gen_ab(x_a).cpu().detach()
                x_ba = self.gen_ba(x_b, n_gen).cpu().detach()
                self.gen_stack.append((x_ab, x_ba))

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

    def resume(self, checkpoint_dir):
        # Load generators
        model_names = sorted(glob.glob(os.path.join(checkpoint_dir, 'gen_*.pt')))
        if len(model_names) == 0:
            return 0
        last_model_name = model_names[-1]
        state_dict = torch.load(last_model_name)
        self.gen_ab.load_state_dict(state_dict['a'])
        self.gen_ba.load_state_dict(state_dict['b'])
        self.estimator_noise.load_state_dict(state_dict['e'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        model_names = sorted(glob.glob(os.path.join(checkpoint_dir, 'dis_*.pt')))
        if len(model_names) == 0:
            return 0
        last_model_name = model_names[-1]
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, checkpoint_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save({'a': self.gen_ab.state_dict(),
                    'b': self.gen_ba.state_dict(),
                    'e': self.estimator_noise.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


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
            print(ex)
            continue
