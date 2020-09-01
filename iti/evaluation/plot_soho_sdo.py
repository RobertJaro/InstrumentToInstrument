import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from sunpy.cm import cm
from torch.utils.data import DataLoader

from iti.data.dataset import SOHODataset
from iti.data.editor import PaddingEditor
from iti.train.model import DiscriminatorMode
from iti.train.trainer import Trainer

sdo_shape = 2048
soho_shape = 1024
base_path = "/gss/r.jarolim/prediction/iti/soho_sdo_v15"

os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)
soho_dataset = SOHODataset("/gss/r.jarolim/data/soho/valid")
soho_dataset.addEditor(PaddingEditor((soho_shape, soho_shape)))
loader = DataLoader(soho_dataset, batch_size=1, shuffle=True)
iter = loader.__iter__()

trainer = Trainer(5, 5, upsampling=1, discriminator_mode=DiscriminatorMode.CHANNELS, norm='in_rs', lambda_diversity=0)
trainer.cuda()
iteration = trainer.resume(base_path)

with torch.no_grad():
    soho_img = next(iter).float().cuda()
    sdo_img = trainer.forwardAB(soho_img)
    soho_img = soho_img.detach().cpu().numpy()
    sdo_img = sdo_img.detach().cpu().numpy()

soho_cmaps = [
    cm.sohoeit171,
    cm.sohoeit195,
    cm.sohoeit284,
    cm.sohoeit304,
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
    imsave(os.path.join(base_path, 'evaluation/full_disc_soho_%d_%d.jpg' % (c, iteration)),
           soho_cmaps[c](plt.Normalize(vmin=-1, vmax=1)(soho_img[0, c]))[..., :3])
    imsave(os.path.join(base_path, 'evaluation/full_disc_sdo_%d_%d.jpg' % (c, iteration)),
           sdo_cmaps[c](plt.Normalize(vmin=-1, vmax=1)(sdo_img[0, c]))[..., :3])
