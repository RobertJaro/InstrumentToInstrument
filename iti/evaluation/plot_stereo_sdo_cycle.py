import os

from iti.data.editor import RandomPatchEditor, SliceEditor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from sunpy.visualization.colormaps import cm
from torch.utils.data import DataLoader

from iti.data.dataset import SDODataset
from iti.train.model import DiscriminatorMode
from iti.train.trainer import Trainer

import numpy as np

base_path = "/gss/r.jarolim/iti/stereo_v5"

os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)
sdo_valid = SDODataset("/gss/r.jarolim/data/sdo/valid", patch_shape=(1024, 1024), resolution=4096)
sdo_valid.addEditor(SliceEditor(0, -1))
sdo_valid.addEditor(RandomPatchEditor((512, 512)))
loader = DataLoader(sdo_valid, batch_size=1, shuffle=True)
iter = loader.__iter__()

trainer = Trainer(4, 4, upsampling=2, discriminator_mode=DiscriminatorMode.CHANNELS, lambda_diversity=0,
                  norm='in_rs_aff')
trainer.cuda()
iteration = trainer.resume(base_path)
trainer.eval()

stereo_cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
]

sdo_cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
]

with torch.no_grad():
    sdo_img = next(iter).float().cuda()
    stereo_img, gen_sdo_img = trainer.forwardBAB(sdo_img)
    stereo_img = stereo_img.detach().cpu().numpy()
    sdo_img = sdo_img.detach().cpu().numpy()
    gen_sdo_img = gen_sdo_img.detach().cpu().numpy()


fig, axs = plt.subplots(1, 4, figsize=(4 * 4, 1 * 4))
[ax.set_axis_off() for ax in np.ravel(axs)]
for c in range(4):
    axs[c].imshow(sdo_img[0, c], vmin=-1, vmax=1, cmap=sdo_cmaps[c])

plt.tight_layout(0)
plt.savefig(os.path.join(base_path, 'evaluation/cycle_%06d_sdo.jpg' % iteration), dpi=300)
plt.close()

fig, axs = plt.subplots(1, 4, figsize=(4 * 4, 1 * 4))
[ax.set_axis_off() for ax in np.ravel(axs)]
for c in range(4):
    axs[c].imshow(stereo_img[0, c], vmin=-1, vmax=1, cmap=stereo_cmaps[c])

plt.tight_layout(0)
plt.savefig(os.path.join(base_path, 'evaluation/cycle_%06d_stereo.jpg' % iteration), dpi=300)
plt.close()

fig, axs = plt.subplots(1, 4, figsize=(4 * 4, 1 * 4))
[ax.set_axis_off() for ax in np.ravel(axs)]
for c in range(4):
    axs[c].imshow(gen_sdo_img[0, c], vmin=-1, vmax=1, cmap=sdo_cmaps[c])

plt.tight_layout(0)
plt.savefig(os.path.join(base_path, 'evaluation/cycle_%06d_gen_sdo.jpg' % iteration), dpi=300)
plt.close()
