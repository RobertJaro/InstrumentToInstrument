import glob
import os
import shutil

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from itipy.data.dataset import HinodeDataset

base_path = '/gpfs/gpfs0/robert.jarolim/data/converted/hinode_continuum_img'
os.makedirs(base_path, exist_ok=True)

hinode_files = glob.glob('/gpfs/gpfs0/robert.jarolim/data/iti/hinode_iti2022_prep/*.fits')
hinode_train = HinodeDataset(hinode_files)

loader = DataLoader(hinode_train, batch_size=1, shuffle=False, num_workers=4)

for d, file in tqdm(zip(loader, hinode_files)):
    img_path = os.path.join(base_path, os.path.basename(file).replace('fits', 'jpg'))
    if os.path.exists(img_path):
        continue
    plt.figure(figsize=(2, 2))
    plt.imshow(d[0, 0], cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img_path)
    plt.close()

shutil.make_archive('/gpfs/gpfs0/robert.jarolim/data/converted/hinode_continuum_img', 'zip', '/gpfs/gpfs0/robert.jarolim/data/converted/hinode_continuum_img')