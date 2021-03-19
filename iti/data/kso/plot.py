import os
import shutil

from torch.utils.data import DataLoader
from tqdm import tqdm

from iti.data.dataset import KSOFlatDataset

from matplotlib import pyplot as plt

path = '/gss/r.jarolim/data/converted/kso_synoptic_img'
dataset = KSOFlatDataset("/gss/r.jarolim/data/kso_synoptic", 1024)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12)

for bn, img in tqdm(zip(dataset.data, loader), total=len(loader)):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img[0, 0], vmin=-1, vmax=1, cmap='gray')
    ax.set_axis_off()
    plt.tight_layout(0)
    fig.savefig(os.path.join(path, os.path.basename(bn).replace('.fts.gz', '.jpg')), dpi=100)
    plt.close(fig)

shutil.make_archive('/gss/r.jarolim/data/converted/kso_synoptic_img', 'zip', '/gss/r.jarolim/data/converted/kso_synoptic_img')