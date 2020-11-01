import os

from iti.data.editor import RandomPatchEditor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from torch.utils.data import DataLoader

from iti.data.dataset import KSOFilmDataset, StorageDataset
from iti.train.trainer import Trainer

base_path = "/gss/r.jarolim/iti/film_v4"

os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)
film_dataset = KSOFilmDataset("/gss/r.jarolim/data/filtered_kso_plate", 512)
film_dataset = StorageDataset(film_dataset, '/gss/r.jarolim/data/converted/iti/kso_film_512', ext_editors=[RandomPatchEditor((256, 256))])

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

for i in range(len(film_img)):
    imsave(os.path.join(base_path, 'evaluation/film_%d_%d.jpg' % (iteration, i)),
           plt.Normalize(vmin=-1, vmax=1)(film_img[i, 0]))
    imsave(os.path.join(base_path, 'evaluation/restored_%d_%d.jpg' % (iteration, i)),
           plt.Normalize(vmin=-1, vmax=1)(restored_img[i, 0]))
