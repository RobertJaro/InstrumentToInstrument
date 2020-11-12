import os

from iti.data.editor import RandomPatchEditor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from sunpy.cm import cm
from torch.utils.data import DataLoader

from iti.data.dataset import SDODataset
from iti.train.model import DiscriminatorMode
from iti.train.trainer import Trainer

import numpy as np

soho_shape = 128
base_path = "/gss/r.jarolim/iti/soho_sdo_v23"

os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)
sdo_valid = SDODataset("/gss/r.jarolim/data/sdo/valid", patch_shape=(512, 512))
sdo_valid.addEditor(RandomPatchEditor((256, 256)))
loader = DataLoader(sdo_valid, batch_size=1, shuffle=True)
iter = loader.__iter__()

trainer = Trainer(5, 5, upsampling=1, discriminator_mode=DiscriminatorMode.CHANNELS,
                  lambda_diversity=0, norm='in_rs_aff')
trainer.cuda()
iteration = trainer.resume(base_path)
trainer.eval()

soho_cmaps = [
    # cm.sohoeit171,
    # cm.sohoeit195,
    # cm.sohoeit284,
    # cm.sohoeit304,
    # plt.get_cmap('gray')
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    plt.get_cmap('gray')
]

sdo_cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    plt.get_cmap('gray')
]

with torch.no_grad():
    sdo_img = next(iter).float().cuda()
    soho_img, gen_sdo_img = trainer.forwardBAB(sdo_img)
    soho_img = soho_img.detach().cpu().numpy()
    sdo_img = sdo_img.detach().cpu().numpy()
    gen_sdo_img = gen_sdo_img.detach().cpu().numpy()

soho_img[0, -1] = -soho_img[0, -1]

fig, axs = plt.subplots(1, 5, figsize=(5 * 4, 1 * 4))
[ax.set_axis_off() for ax in np.ravel(axs)]
for c in range(5):
    axs[c].imshow(sdo_img[0, c], vmin=-1, vmax=1, cmap=sdo_cmaps[c])

plt.tight_layout()
plt.savefig(os.path.join(base_path, 'evaluation/cycle_sdo_%06d.jpg' % iteration), dpi=300)
plt.close()

fig, axs = plt.subplots(1, 5, figsize=(5 * 4, 1 * 4))
[ax.set_axis_off() for ax in np.ravel(axs)]
for c in range(5):
    axs[c].imshow(soho_img[0, c], vmin=-1, vmax=1, cmap=soho_cmaps[c])

plt.tight_layout()
plt.savefig(os.path.join(base_path, 'evaluation/cycle_soho_%06d.jpg' % iteration), dpi=300)
plt.close()

fig, axs = plt.subplots(1, 5, figsize=(5 * 4, 1 * 4))
[ax.set_axis_off() for ax in np.ravel(axs)]
for c in range(5):
    axs[c].imshow(gen_sdo_img[0, c], vmin=-1, vmax=1, cmap=sdo_cmaps[c])

plt.tight_layout()
plt.savefig(os.path.join(base_path, 'evaluation/cycle_gen_sdo_%06d.jpg' % iteration), dpi=300)
plt.close()
