import os

from astropy.coordinates import SkyCoord
from matplotlib.colors import Normalize
from sunpy.map import Map

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from sunpy.visualization.colormaps import cm
from torch.utils.data import DataLoader

from iti.data.dataset import SOHODataset
from iti.data.editor import PaddingEditor, LoadMapEditor, NormalizeRadiusEditor
from iti.train.model import DiscriminatorMode
from iti.train.trainer import Trainer

from astropy import units as u

import numpy as np

sdo_shape = 2048
soho_shape = 1024
base_path = "/gss/r.jarolim/iti/soho_sdo_v23"

os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)
soho_dataset = SOHODataset("/gss/r.jarolim/data/soho/valid", basenames=['2000-11-12T01:19.fits'])
soho_dataset.addEditor(PaddingEditor((soho_shape, soho_shape)))
loader = DataLoader(soho_dataset, batch_size=1)
iter = loader.__iter__()

trainer = Trainer(5, 5, upsampling=1, discriminator_mode=DiscriminatorMode.CHANNELS,
                  lambda_diversity=0, norm='in_rs_aff')
trainer.cuda()
iteration = trainer.resume(base_path)
trainer.eval()

with torch.no_grad():
    soho_img = next(iter).float().cuda()
    sdo_img = trainer.forwardAB(soho_img)
    soho_img = soho_img.detach().cpu().numpy()
    sdo_img = sdo_img.detach().cpu().numpy()

sdo_img[0, -1] = -sdo_img[0, -1]

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

reference_map, _ = LoadMapEditor().call(soho_dataset.data_sets[0].data[0])
reference_map = NormalizeRadiusEditor(soho_shape).call(reference_map)

fig, axs = plt.subplots(2, 5, figsize=(5 * 4, 2 * 4), sharex=True, sharey=True)
sub_frame_x = [-500, -350] * u.arcsec
sub_frame_y = [-450, -300] * u.arcsec

for c in range(5):
    s_map = Map(soho_img[0, c], reference_map.meta)
    s_map = s_map.submap(SkyCoord(sub_frame_x, sub_frame_y, frame=s_map.coordinate_frame))
    s_map.plot(axes=axs[0, c], cmap=soho_cmaps[c], title='', norm=Normalize(vmin=-1, vmax=1))

reference_map = reference_map.resample(sdo_img.shape[2:] * u.pix)
for c in range(5):
    s_map = Map(sdo_img[0, c], reference_map.meta)
    s_map = s_map.submap(SkyCoord(sub_frame_x, sub_frame_y, frame=s_map.coordinate_frame))
    s_map.plot(axes=axs[1, c], cmap=sdo_cmaps[c], title='', norm=Normalize(vmin=-1, vmax=1))

[(ax.set_xlabel(None), ax.set_ylabel(None)) for ax in np.ravel(axs)]
fontsize = 13
axs[0, 0].set_ylabel('Helioprojective latitude [arcsec]', fontsize=fontsize)
axs[1, 0].set_ylabel('Helioprojective latitude [arcsec]', fontsize=fontsize)
[ax.set_xlabel('Helioprojective longitude [arcsec]', fontsize=fontsize) for ax in axs[1]]

plt.tight_layout()
plt.savefig(os.path.join(base_path, 'evaluation/comparison_%06d.jpg' % iteration), dpi=300)
plt.close()