import glob
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from iti.data.dataset import HMIContinuumDataset
from iti.data.editor import sdo_norms
from iti.data.loader import HMIContinuumLoader

files = glob.glob('/gpfs/gpfs0/robert.jarolim/data/iti/hmi_continuum/*.fits')
shuffle(files)
loader = HMIContinuumLoader()
# loader = DataLoader(hmi_dataset, batch_size=1, num_workers=4)
# iter = loader.__iter__()

means = []
stds = []
norm = sdo_norms['continuum']
for f in files:
    data = loader(f)
    data = data.data
    means += [np.nanmean(data)]
    stds += [np.nanstd(data)]
    print('--------------------')
    print(f'Mean: {np.mean(means)}')
    print(f'Std: {np.mean(stds)}')
    plt.imsave('/beegfs/home/robert.jarolim/iti_evaluation/test.jpg', data, cmap='gray')
