import os
import random

from matplotlib import pyplot as plt
from sunpy.visualization.colormaps import cm

from iti.data.dataset import SECCHIDataset, EITDataset, AIADataset, SDODataset

base_path = '/gss/r.jarolim/data/ch_detection'
channels = [
    171,
    193,
    211,
    304,
]
cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
]

sdo_ds = SDODataset(base_path)

for c, cmap in zip(channels, cmaps):
    ds = AIADataset(os.path.join(base_path, '%d' % c), c)
    data = ds[random.randint(0, len(ds) -  1)]
    plt.figure(figsize=(6, 6))
    plt.imshow(data[0], vmin=-1, vmax=1, cmap=cmap)
    plt.axis('off')
    plt.tight_layout(0)
    plt.savefig('/gss/r.jarolim/data/converted/iti_samples/sdo_%d.jpg' % c, dpi=300)
    plt.close()
