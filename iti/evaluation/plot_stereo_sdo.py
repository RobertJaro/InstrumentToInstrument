import os

from astropy.coordinates import SkyCoord
from matplotlib.colors import Normalize
from sunpy.map import Map

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from sunpy.cm import cm
from torch.utils.data import DataLoader

from iti.data.dataset import STEREODataset
from iti.data.editor import PaddingEditor, LoadMapEditor, NormalizeRadiusEditor
from iti.train.model import DiscriminatorMode
from iti.train.trainer import Trainer

import numpy as np
from astropy import units as u

soho_shape = 1024
base_path = "/gss/r.jarolim/iti/stereo_v5"

os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)
stereo_dataset = STEREODataset("/gss/r.jarolim/data/stereo_prep/valid", base_names=['2016-11-01T19:46.fits'])
stereo_dataset.addEditor(PaddingEditor((soho_shape, soho_shape)))
loader = DataLoader(stereo_dataset, batch_size=1)
iter = loader.__iter__()

trainer = Trainer(4, 4, upsampling=2, discriminator_mode=DiscriminatorMode.CHANNELS, lambda_diversity=0,
                  norm='in_rs_aff')
trainer.cuda()
iteration = trainer.resume(base_path)
trainer.eval()

with torch.no_grad():
    stereo_img = next(iter).float().cuda()
    sdo_img = trainer.forwardAB(stereo_img)
    stereo_img = stereo_img.detach().cpu().numpy()
    sdo_img = sdo_img.detach().cpu().numpy()

stereo_cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304
]

sdo_cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304
]

reference_map, _ = LoadMapEditor().call(stereo_dataset.data_sets[0].data[0])
reference_map = NormalizeRadiusEditor(soho_shape).call(reference_map)

fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
sub_frame_x = [-300, 100] * u.arcsec
sub_frame_y = [350, -50] * u.arcsec

for c in range(4):
    s_map = Map(stereo_img[0, c], reference_map.meta)
    s_map = s_map.submap(SkyCoord(sub_frame_x, sub_frame_y, frame=s_map.coordinate_frame))
    s_map.plot(axes=axs[0, c], cmap=stereo_cmaps[c], norm=Normalize(vmin=-1, vmax=1), title='')

reference_map = reference_map.resample(sdo_img.shape[2:] * u.pix)
for c in range(4):
    s_map = Map(sdo_img[0, c], reference_map.meta)
    s_map = s_map.submap(SkyCoord(sub_frame_x, sub_frame_y, frame=s_map.coordinate_frame))
    s_map.plot(axes=axs[1, c], cmap=sdo_cmaps[c], norm=Normalize(vmin=-1, vmax=1), title='')


[(ax.set_xlabel(None), ax.set_ylabel(None)) for ax in np.ravel(axs)]
fontsize = 13
axs[0,0].set_ylabel('Helioprojective latitude [arcsec]', fontsize=fontsize)
axs[1,0].set_ylabel('Helioprojective latitude [arcsec]', fontsize=fontsize)
[ax.set_xlabel('Helioprojective longitude [arcsec]', fontsize=fontsize) for ax in axs[1]]

plt.tight_layout(0.1)
plt.savefig(os.path.join(base_path, 'evaluation/comparison_%06d.jpg' % iteration), dpi=300)
plt.close()
