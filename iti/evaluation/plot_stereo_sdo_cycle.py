import os

from matplotlib.colors import Normalize
from skimage.io import imsave

from iti.data.editor import RandomPatchEditor, SliceEditor, sdo_norms, BrightestPixelPatchEditor



import torch
from matplotlib import pyplot as plt
from sunpy.visualization.colormaps import cm
from torch.utils.data import DataLoader

from iti.data.dataset import SDODataset, STEREODataset
from iti.train.model import DiscriminatorMode
from iti.trainer import Trainer

import numpy as np

base_path = "/gpfs/gpfs0/robert.jarolim/iti/stereo_to_sdo_v1"
evaluation_path = os.path.join(base_path, 'evaluation')
os.makedirs(evaluation_path, exist_ok=True)
sdo_valid = SDODataset("/gpfs/gpfs0/robert.jarolim/data/iti/sdo", resolution=4096, months=[11, 12])
sdo_valid.addEditor(SliceEditor(0, -1))
loader = DataLoader(sdo_valid, batch_size=1, shuffle=False)
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

cmap = cm.sdoaia171
norm = Normalize(vmin=-1, vmax=1)
imsave(os.path.join(evaluation_path, 'cycle_sdo.jpg'), cmap(norm(sdo_img)[0, 0])[..., :-1], check_contrast=False)
imsave(os.path.join(evaluation_path, 'cycle_stereo.jpg'), cmap(norm(stereo_img)[0, 0])[..., :-1], check_contrast=False)
imsave(os.path.join(evaluation_path, 'cycle_iti.jpg'), cmap(norm(gen_sdo_img)[0, 0])[..., :-1], check_contrast=False)

stereo_ds = STEREODataset("/gpfs/gpfs0/robert.jarolim/data/iti/stereo_iti2021_prep", months=[11, 12])
img = stereo_ds[10][0]
imsave(os.path.join(evaluation_path, 'ref_stereo.jpg'), cmap(norm(img))[..., :-1], check_contrast=False)

fig, axs = plt.subplots(1, 4, figsize=(4 * 4, 1 * 4))
[ax.set_axis_off() for ax in np.ravel(axs)]
for c in range(4):
    axs[c].imshow(sdo_img[0, c], vmin=-1, vmax=1, cmap=sdo_cmaps[c])

plt.tight_layout(pad=0)
plt.savefig(os.path.join(base_path, 'evaluation/cycle_%06d_sdo.jpg' % iteration), dpi=300)
plt.close()

fig, axs = plt.subplots(1, 4, figsize=(4 * 4, 1 * 4))
[ax.set_axis_off() for ax in np.ravel(axs)]
for c in range(4):
    axs[c].imshow(stereo_img[0, c], vmin=-1, vmax=1, cmap=stereo_cmaps[c])

plt.tight_layout(pad=0)
plt.savefig(os.path.join(base_path, 'evaluation/cycle_%06d_stereo.jpg' % iteration), dpi=300)
plt.close()

fig, axs = plt.subplots(1, 4, figsize=(4 * 4, 1 * 4))
[ax.set_axis_off() for ax in np.ravel(axs)]
for c in range(4):
    axs[c].imshow(gen_sdo_img[0, c], vmin=-1, vmax=1, cmap=sdo_cmaps[c])

plt.tight_layout(pad=0)
plt.savefig(os.path.join(base_path, 'evaluation/cycle_%06d_gen_sdo.jpg' % iteration), dpi=300)
plt.close()
