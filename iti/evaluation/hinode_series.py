import glob
import os
import shutil
from random import sample

from astropy.io.fits import getheader, getdata
from tqdm import tqdm

from iti.data.dataset import HMIContinuumDataset, StorageDataset, HinodeDataset
from iti.data.editor import NanEditor, RandomPatchEditor
from matplotlib import  pyplot as plt
import numpy as np

# hinode_dataset = HinodeDataset("/gss/r.jarolim/data/hinode/level1")
# ds = StorageDataset(hinode_dataset, '/gss/r.jarolim/data/converted/hinode_train', ext_editors=[NanEditor()])
#
# for f, data in tqdm(zip(hinode_dataset.data, ds)):
#     plt.imshow(data[0], vmin=-1, vmax=1)
#     plt.colorbar()
#     plt.savefig('/gss/r.jarolim/data/hinode/imgs/%s.jpg' % os.path.basename(f), dpi=80)
#     plt.close()
#
# shutil.make_archive('/gss/r.jarolim/data/hinode/imgs', 'zip', '/gss/r.jarolim/data/hinode/imgs')

files = glob.glob('/gss/r.jarolim/data/hinode/level1/*.fits')
special_features = []
quite_sun = []
for f in tqdm(files[::10]):
    data = np.ravel(getdata(f)) / getheader(f)['EXPTIME']
    if np.sum(data < 25000) > 2000:
        special_features.append(f)
    else:
        quite_sun.append(f)

print(len(special_features), len(quite_sun))
