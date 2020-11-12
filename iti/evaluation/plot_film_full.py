import os

from iti.data.editor import RandomPatchEditor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from torch.utils.data import DataLoader

from iti.data.dataset import KSOFilmDataset, StorageDataset
from iti.train.trainer import Trainer

import numpy as np

base_path = "/gss/r.jarolim/iti/film_v6"

os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)
film_dataset = KSOFilmDataset("/gss/r.jarolim/data/filtered_kso_plate", 512)
film_dataset.addEditor(RandomPatchEditor((256, 256)))
#film_dataset = StorageDataset(film_dataset, '/gss/r.jarolim/data/converted/iti/kso_film_512', ext_editors=[])

loader = DataLoader(film_dataset, batch_size=5, shuffle=True)
iter = loader.__iter__()

trainer = Trainer(1, 1, norm='in_rs_aff')
trainer.cuda()
iteration = trainer.resume(base_path)
trainer.eval()

with torch.no_grad():
    film_img = next(iter).float().cuda()
    restored_img = trainer.forwardAB(film_img)
    film_img = film_img.detach().cpu().numpy()
    restored_img = restored_img.detach().cpu().numpy()

fig, axs = plt.subplots(film_img.shape[0], 2, figsize=(3 * 2, 3 * film_img.shape[0]))
[ax.set_axis_off() for ax in np.ravel(axs)]
for i in range(film_img.shape[0]):
    axs[i, 0].imshow(film_img[i, 0], cmap='gray', vmin=-1, vmax=1)
    axs[i, 1].imshow(restored_img[i, 0], cmap='gray', vmin=-1, vmax=1)

plt.tight_layout()
plt.savefig(os.path.join(base_path, 'evaluation/restoration_%d.jpg' % iteration), dpi=300)
plt.close()