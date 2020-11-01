import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from sunpy.cm import cm
from torch.utils.data import DataLoader

from iti.data.dataset import SOHODataset, KSODataset
from iti.data.editor import PaddingEditor
from iti.train.model import DiscriminatorMode
from iti.train.trainer import Trainer

resolution = 512
base_path = "/gss/r.jarolim/iti/kso_quality_512_v1"
os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)

q2_dataset = KSODataset("/gss/r.jarolim/data/kso_general/quality2", resolution)
loader = DataLoader(q2_dataset, batch_size=4, shuffle=True)
iter = loader.__iter__()

trainer = Trainer(1, 1, norm='in_aff')
trainer.cuda()
iteration = trainer.resume(base_path, 140000)

with torch.no_grad():
    q2_img = next(iter).float().cuda()
    q1_img = trainer.forwardAB(q2_img)
    q2_img = q2_img.detach().cpu().numpy()
    q1_img = q1_img.detach().cpu().numpy()

for i in range(4):
    imsave(os.path.join(base_path, 'evaluation/q2_%d_%d.jpg' % (iteration, i)), plt.Normalize(vmin=-1, vmax=1)(q2_img[i, 0]))
    imsave(os.path.join(base_path, 'evaluation/q1_%d_%d.jpg' % (iteration, i)), plt.Normalize(vmin=-1, vmax=1)(q1_img[i, 0]))
