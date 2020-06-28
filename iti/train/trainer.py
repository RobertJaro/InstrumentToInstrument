import glob
import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from iti.train.model import GeneratorAB, GeneratorBA, Discriminator, NoiseEstimator


class Trainer(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, noise_dim, depth=3, learning_rate=1e-4):
        super().__init__()
        self.noise_dim= noise_dim
        self.depth = depth
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b

        # Initiate the networks
        self.gen_ab = GeneratorAB(input_dim_a, input_dim_b, depth, depth)  # generator for domain a
        self.gen_ba = GeneratorBA(input_dim_a, input_dim_b, noise_dim, depth, depth)  # generator for domain a
        self.dis_a = Discriminator(input_dim_a)  # discriminator for domain a
        self.dis_b = Discriminator(input_dim_b)  # discriminator for domain b
        self.estimator_noise = NoiseEstimator(input_dim_a, depth, 64, noise_dim)

        # Setup the optimizers
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_ab.parameters()) + list(self.gen_ba.parameters()) + list(self.estimator_noise.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=learning_rate, betas=(0.5, 0.999))
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=learning_rate, betas=(0.5, 0.999))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        n_gen = Variable(torch.randn(x_b.size(0), self.noise_dim, x_b.size(2) // 2 ** self.depth,
                            x_b.size(3) // 2 ** self.depth).cuda())
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
        n_gen = Variable(torch.randn(x_b.size(0), self.noise_dim, x_b.size(2) // 2 ** self.depth, x_b.size(3) // 2 ** self.depth).cuda())
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
        n_gen = Variable(torch.randn(x_b.size(0), self.noise_dim, x_b.size(2) // 2 ** self.depth, x_b.size(3) // 2 ** self.depth).cuda())
        x_ba = self.gen_ba(x_b, n_gen)
        x_bab = self.gen_ab(x_ba)
        self.train()
        return x_ba, x_bab

    def generator_update(self, x_a, x_b):
        self.gen_opt.zero_grad()

        # noise init
        n_a = self.estimator_noise(x_a)
        n_gen = Variable(torch.randn(*n_a.size()).cuda())

        # identity
        x_b_identity = self.gen_ab(x_b)
        x_a_identity = self.gen_ba(x_a, n_a)
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
        self.loss_gen_a_content = self.dis_a.calc_content_loss(x_a, x_aba)
        self.loss_gen_b_content = self.dis_b.calc_content_loss(x_b, x_bab)
        self.loss_gen_a_identity_content = self.dis_a.calc_content_loss(x_a, x_a_identity)
        self.loss_gen_b_identity_content = self.dis_b.calc_content_loss(x_b, x_b_identity)
        # Noise loss
        self.loss_gen_ba_noise = self.recon_criterion(n_ba, n_gen)
        self.loss_gen_aba_noise = self.recon_criterion(n_aba, n_a)
        self.loss_gen_a_identity_noise = self.recon_criterion(n_a_identity, n_a)
        # total loss
        self.loss_gen_total = 0 * (self.loss_gen_a_identity + self.loss_gen_b_identity) + \
                              0 * (self.loss_gen_a_translate + self.loss_gen_b_translate) + \
                              self.loss_gen_adv_a + self.loss_gen_adv_b + \
                              self.loss_gen_a_content + self.loss_gen_b_content + \
                              self.loss_gen_a_identity_content + self.loss_gen_b_identity_content +\
                              self.loss_gen_ba_noise + self.loss_gen_aba_noise + self.loss_gen_a_identity_noise
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def discriminator_update(self, x_a, x_b):
        self.dis_opt.zero_grad()
        # noise
        n_gen = Variable(torch.randn(x_b.size(0), self.noise_dim, x_b.size(2) // 2 ** self.depth, x_b.size(3) // 2 ** self.depth).cuda())
        # translate
        with torch.no_grad():
            x_ab = self.gen_ab(x_a)
            x_ba = self.gen_ba(x_b, n_gen)

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = self.loss_dis_a + self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def resume(self, checkpoint_dir):
        # Load generators
        model_names = sorted(glob.glob(os.path.join(checkpoint_dir, 'gen_*.pt')))
        if len(model_names) == 0:
            return 0
        last_model_name = model_names[-1]
        state_dict = torch.load(last_model_name)
        self.gen_ab.load_state_dict(state_dict['a'])
        self.gen_ba.load_state_dict(state_dict['b'])
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
        torch.save({'a': self.gen_ab.state_dict(), 'b': self.gen_ba.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


def convertSet(data_set, store_path):
    loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=16)
    for i, sample in enumerate(loader):
        np.save(os.path.join(store_path, '%d.npy' % i), sample[0])

def loop(iterable):
    it = iterable.__iter__()

    while True:
        try:
            yield it.next()
        except StopIteration:
            it = iterable.__iter__()
            yield it.next()