import os

from astropy.coordinates import SkyCoord
from matplotlib.colors import Normalize
from sunpy.map import Map

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from sunpy.visualization.colormaps import cm
from torch.utils.data import DataLoader

from iti.data.dataset import SOHODataset, STEREODataset
from iti.data.editor import PaddingEditor, LoadMapEditor, NormalizeRadiusEditor
from iti.train.model import DiscriminatorMode
from iti.train.trainer import Trainer

from astropy import units as u
import numpy as np

stereo_shape = 1024
base_path = "/gss/r.jarolim/iti/stereo_mag_v7"

os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)
stereo_dataset = STEREODataset("/gss/r.jarolim/data/stereo_prep/train", basenames=['2007-01-01T05:03.fits'])
stereo_dataset.addEditor(PaddingEditor((stereo_shape, stereo_shape)))
loader = DataLoader(stereo_dataset, batch_size=1)
iter = loader.__iter__()

soho_dataset = SOHODataset("/gss/r.jarolim/data/soho/train", basenames=['2007-01-01T01:19.fits'])
soho_dataset.addEditor(PaddingEditor((stereo_shape, stereo_shape)))

trainer = Trainer(4, 5, upsampling=2, discriminator_mode=DiscriminatorMode.CHANNELS, lambda_diversity=0, norm='in_rs_aff')
trainer.cuda()
iteration = trainer.resume(base_path)

with torch.no_grad():
    stereo_img = next(iter).float().cuda()
    iti_img = trainer.forwardAB(stereo_img)
    stereo_img = stereo_img.detach().cpu().numpy()
    iti_img = iti_img.detach().cpu().numpy()
    soho_img = np.array([soho_dataset[0]])

soho_img[0, 4] = np.flip(np.flip(soho_img[0, 4], 0), 1)
soho_img[0, 4] = np.abs(soho_img[0, 4])
iti_img[0, 4] = (iti_img[0, 4] + 1) / 2

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
    cm.sdoaia304,
    plt.get_cmap('gray')
]

reference_map, _ = LoadMapEditor().call(stereo_dataset.data_sets[0].data[0])
reference_map = NormalizeRadiusEditor(stereo_shape).call(reference_map)

fig, axs = plt.subplots(3, 5, figsize=(20, 12), sharex=True, sharey=True)
sub_frame_x = [-300, 100] * u.arcsec
sub_frame_y = [-500, -100] * u.arcsec

for c in range(4):
    s_map = Map(stereo_img[0, c], reference_map.meta)
    #s_map = s_map.submap(SkyCoord(sub_frame_x, sub_frame_y, frame=s_map.coordinate_frame))
    title = 'STEREO' if c == 0 else ''
    s_map.plot(axes=axs[0, c], cmap=stereo_cmaps[c], norm=Normalize(vmin=-1, vmax=1), title=title)

axs[0, 4].set_axis_off()

reference_map = reference_map.resample(iti_img.shape[2:] * u.pix)
for c in range(5):
    s_map = Map(iti_img[0, c], reference_map.meta)
    #s_map = s_map.submap(SkyCoord(sub_frame_x, sub_frame_y, frame=s_map.coordinate_frame))
    norm = Normalize(vmin=-.5, vmax=.5) if c == 4 else Normalize(vmin=-1, vmax=1)
    title = 'ITI' if c == 0 else ''
    s_map.plot(axes=axs[1, c], cmap=sdo_cmaps[c], norm=norm, title=title)

reference_map = reference_map.resample(soho_img.shape[2:] * u.pix)
for c in range(5):
    s_map = Map(soho_img[0, c], reference_map.meta)
    #s_map = s_map.submap(SkyCoord(sub_frame_x, sub_frame_y, frame=s_map.coordinate_frame))
    norm = Normalize(vmin=-.5, vmax=.5) if c == 4 else Normalize(vmin=-1, vmax=1)
    title = 'SOHO' if c == 0 else ''
    s_map.plot(axes=axs[2, c], cmap=sdo_cmaps[c], norm=norm, title=title)

[(ax.set_xlabel(None), ax.set_ylabel(None)) for ax in np.ravel(axs)]
fontsize = 13
axs[0,0].set_ylabel('Helioprojective latitude [arcsec]', fontsize=fontsize)
axs[1,0].set_ylabel('Helioprojective latitude [arcsec]', fontsize=fontsize)
axs[2,0].set_ylabel('Helioprojective latitude [arcsec]', fontsize=fontsize)
[ax.set_xlabel('Helioprojective longitude [arcsec]', fontsize=fontsize) for ax in axs[2]]

plt.tight_layout(0.1)
plt.savefig(os.path.join(base_path, 'evaluation/20070101_%06d.jpg' % iteration), dpi=300)
plt.close()
