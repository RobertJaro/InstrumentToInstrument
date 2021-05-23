import os

from skimage.io import imsave

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from sunpy.visualization.colormaps import cm
from torch.utils.data import DataLoader

from iti.data.dataset import STEREODataset
from iti.data.editor import PaddingEditor
from iti.train.model import DiscriminatorMode
from iti.train.trainer import Trainer

soho_shape = 1024
base_path = "/gss/r.jarolim/iti/stereo_v5"

os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)
stereo_dataset = STEREODataset("/gss/r.jarolim/data/stereo_prep/valid", basenames=['2016-11-01T19:46.fits'])
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

for c in range(4):
    imsave(os.path.join(base_path, 'evaluation/full_disc_%d_%d_stereo.jpg' % (iteration, c)),
           stereo_cmaps[c](plt.Normalize(vmin=-1, vmax=1)(stereo_img[0, c]))[..., :3])
    imsave(os.path.join(base_path, 'evaluation/full_disc_%d_%d_sdo.jpg' % (iteration, c)),
           sdo_cmaps[c](plt.Normalize(vmin=-1, vmax=1)(sdo_img[0, c]))[..., :3])
