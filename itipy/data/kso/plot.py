import os
import shutil

from torch.utils.data import DataLoader
from tqdm import tqdm

from itipy.data.dataset import KSOFlatDataset

from matplotlib import pyplot as plt

import numpy as np

path = '/gss/r.jarolim/data/converted/kso_synoptic_img'
os.makedirs(path, exist_ok=True)
dataset = KSOFlatDataset("/gss/r.jarolim/data/kso_synoptic", 1024)
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=12)

values = []
for bn, img in tqdm(zip(dataset.data, loader), total=len(loader)):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    values += list(np.ravel(img))
    ax.imshow(img[0, 0], vmin=-1, vmax=1, cmap='gray')
    ax.set_axis_off()
    plt.tight_layout(0)
    fig.savefig(os.path.join(path, os.path.basename(bn).replace('.fts.gz', '.jpg')), dpi=100)
    plt.close(fig)

plt.hist(values, bins=100)
plt.tight_layout(0)
plt.savefig('/gss/r.jarolim/data/converted/kso_synoptic_img/hist.jpg', dpi=100)
plt.close()
#shutil.make_archive('/gss/r.jarolim/data/converted/kso_synoptic_img', 'zip', '/gss/r.jarolim/data/converted/kso_synoptic_img')