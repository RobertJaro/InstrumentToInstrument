import os



import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from itipy.data.dataset import KSOFlatDataset
from itipy.trainer import Trainer

import numpy as np

resolution = 1024
base_path = "/gss/r.jarolim/iti/kso_quality_1024_v9"
epoch = 360000
os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)

q1_dataset = KSOFlatDataset("/gss/r.jarolim/data/kso_synoptic", resolution)
q1_loader = DataLoader(q1_dataset, batch_size=4, shuffle=True)
q1_iter = q1_loader.__iter__()

trainer = Trainer(1, 1)
trainer.resume(checkpoint_dir=base_path, epoch=epoch)
trainer.cuda()

with torch.no_grad():
    #
    B_img = next(q1_iter).float().cuda()
    BA_imgs = [trainer.forwardBA(B_img).detach().cpu().numpy() for _ in range(6)]
    BA_imgs = np.concatenate(BA_imgs, axis=1)
    #
    B_img = B_img.detach().cpu().numpy()

fig, axs = plt.subplots(BA_imgs.shape[0], BA_imgs.shape[1] + 1, figsize=(3 * (BA_imgs.shape[1] + 1), 3 * BA_imgs.shape[0]))
[ax.set_axis_off() for ax in np.ravel(axs)]
for i in range(BA_imgs.shape[0]):
    axs[i, 0].imshow(B_img[i, 0], cmap='gray', )
    for j in range(BA_imgs.shape[1]):
        axs[i, j + 1].imshow(BA_imgs[i, j], cmap='gray', )

plt.tight_layout()
plt.savefig(os.path.join(base_path, 'evaluation/variation_%d.jpg' % epoch), dpi=300)
plt.close()

os.path.join(base_path, 'generator_AB.pt')