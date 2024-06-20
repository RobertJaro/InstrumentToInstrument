import os

from skimage.io import imsave



import torch
from matplotlib import pyplot as plt
from sunpy.visualization.colormaps import cm
from torch.utils.data import DataLoader

from iti.data.dataset import SOHODataset
from iti.data.editor import PaddingEditor
from iti.train.model import DiscriminatorMode
from iti.trainer import Trainer

sdo_shape = 2048
soho_shape = 1024
base_path = "/gss/r.jarolim/iti/soho_sdo_v23"

os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)
soho_dataset = SOHODataset("/gss/r.jarolim/data/soho/valid", basenames=['2001-12-31T01:19.fits'])
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

for c in range(5):
    imsave(os.path.join(base_path, 'evaluation/full_disc_%d_%d_soho.jpg' % (iteration, c)),
           soho_cmaps[c](plt.Normalize(vmin=-1, vmax=1)(soho_img[0, c]))[..., :3])
    imsave(os.path.join(base_path, 'evaluation/full_disc_%d_%d_sdo.jpg' % (iteration, c)),
           sdo_cmaps[c](plt.Normalize(vmin=-1, vmax=1)(sdo_img[0, c]))[..., :3])
