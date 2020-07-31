import os
import shutil

from tqdm import tqdm

from iti.data.dataset import HMIContinuumDataset, StorageDataset, HinodeDataset
from iti.data.editor import NanEditor, RandomPatchEditor
from matplotlib import  pyplot as plt
import numpy as np
hmi_dataset = HMIContinuumDataset("/gss/r.jarolim/data/hmi_continuum/6173", (256, 256))
ds = StorageDataset(hmi_dataset,
                             '/gss/r.jarolim/data/converted/hmi_train',
                             ext_editors=[])
for f, data in tqdm(zip(hmi_dataset.data, ds)):
    if np.std(data) < 0.01:
        print(np.min(data), np.max(data), np.std(data))
        plt.imshow(data[0], vmin=-1, vmax=1)
        plt.colorbar()
        plt.savefig('/gss/r.jarolim/data/hinode/imgs/%s.jpg' % os.path.basename(f), dpi=80)
        plt.close()

shutil.make_archive('/gss/r.jarolim/data/hinode/imgs', 'zip', '/gss/r.jarolim/data/hinode/imgs')