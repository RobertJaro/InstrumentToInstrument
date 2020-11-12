import os

from iti.data.editor import RandomPatchEditor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from torch.utils.data import DataLoader

from iti.data.dataset import KSODataset
from iti.train.trainer import Trainer

import numpy as np

resolution = 512
base_path = "/gss/r.jarolim/iti/kso_quality_512_v1"
os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)

q2_dataset = KSODataset("/gss/r.jarolim/data/kso_general/quality2", resolution)
#q2_dataset.addEditor(RandomPatchEditor((256, 256)))
q2_loader = DataLoader(q2_dataset, batch_size=4, shuffle=True)
q2_iter = q2_loader.__iter__()

trainer = Trainer(1, 1, norm='in_aff')
trainer.cuda()
iteration = trainer.resume(base_path)

with torch.no_grad():
    A_img = next(q2_iter).float().cuda()
    AB_img = trainer.forwardAB(A_img)
    AB_img = AB_img.detach().cpu().numpy()
    #
    A_img = A_img.detach().cpu().numpy()

fig, axs = plt.subplots(AB_img.shape[0], 2, figsize=(3 * 2, 3 * AB_img.shape[0]))
[ax.set_axis_off() for ax in np.ravel(axs)]
for i in range(AB_img.shape[0]):
    axs[i, 0].imshow(A_img[i, 0], cmap='gray', )
    axs[i, 1].imshow(AB_img[i, 0], cmap='gray', )

plt.tight_layout()
plt.savefig(os.path.join(base_path, 'evaluation/restoration_%d.jpg' % iteration), dpi=300)
plt.close()