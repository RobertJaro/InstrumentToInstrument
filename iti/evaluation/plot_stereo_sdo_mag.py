import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from sunpy.cm import cm
from torch.utils.data import DataLoader

from iti.data.dataset import SOHODataset, STEREODataset
from iti.data.editor import PaddingEditor
from iti.train.model import DiscriminatorMode
from iti.train.trainer import Trainer

sdo_shape = 2048
soho_shape = 1024
base_path = "/gss/r.jarolim/iti/stereo_mag_v2"

os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)
stereo_dataset = STEREODataset("/gss/r.jarolim/data/stereo_prep/valid")
stereo_dataset.addEditor(PaddingEditor((soho_shape, soho_shape)))
loader = DataLoader(stereo_dataset, batch_size=1, shuffle=True)
iter = loader.__iter__()

trainer = Trainer(4, 5, upsampling=2, discriminator_mode=DiscriminatorMode.CHANNELS, lambda_diversity=0)
trainer.cuda()
iteration = trainer.resume(base_path)

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
    cm.sdoaia304,
    plt.get_cmap('gray')
]
for c in range(4):
    imsave(os.path.join(base_path, 'evaluation/full_disc_stereo_%d_%d.jpg' % (iteration, c)),
           stereo_cmaps[c](plt.Normalize(vmin=-1, vmax=1)(stereo_img[0, c]))[..., :3])
for c in range(5):
    imsave(os.path.join(base_path, 'evaluation/full_disc_sdo_%d_%d.jpg' % (iteration, c)),
           sdo_cmaps[c](plt.Normalize(vmin=-1, vmax=1)(sdo_img[0, c]))[..., :3])
