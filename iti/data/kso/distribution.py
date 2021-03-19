import os
import random

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from iti.data.dataset import KSOFlatDataset
from matplotlib import pyplot as plt

import numpy as np

q1_dataset = KSOFlatDataset("/gss/r.jarolim/data/anomaly_data_set/quality1", 512)

fig, axs = plt.subplots(10, 1, figsize=(4, 20), sharex=True)
for i, ax in zip(random.sample(range(len(q1_dataset)), len(axs)), axs):
    ax.hist(np.ravel(q1_dataset[i]), 100)

plt.tight_layout()
plt.savefig('/gss/r.jarolim/iti/kso_quality_512_v4/hist.jpg')

plt.figure(figsize=(8, 4))
maxs = [np.nanmax(q1_dataset[i]) for i in tqdm(random.sample(range(len(q1_dataset)), 100))]
plt.hist(maxs, 50)
plt.tight_layout()
plt.savefig('/gss/r.jarolim/iti/kso_quality_512_v4/hist.jpg')
