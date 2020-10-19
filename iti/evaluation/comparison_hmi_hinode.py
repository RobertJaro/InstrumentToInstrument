import glob
import os
from random import sample

from dateutil.parser import parse

from iti.download.hmi_continuum_download import HMIContinuumFetcher

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from skimage.util import view_as_blocks
from torch.utils.data import DataLoader
from tqdm import tqdm

from iti.data.dataset import HMIContinuumDataset, HinodeDataset
from iti.data.editor import PaddingEditor
from iti.train.trainer import Trainer

hmi_shape = 4096
patch_shape = 1024
n_patches = hmi_shape // patch_shape
base_path = '/gss/r.jarolim/prediction/iti/hmi_hinode_v9'
evaluation_path = os.path.join(base_path, "comparison")
data_path = os.path.join(evaluation_path, "data")
os.makedirs(data_path, exist_ok=True)

hinode_files = glob.glob('/gss/r.jarolim/data/hinode/level1/*.fits')
hinode_sample = sample(hinode_files, 4)
hinode_dates = [parse(f[-22:-7].replace('_', 'T')) for f in hinode_sample]

hinode_dataset = HinodeDataset(hinode_sample)
for f, d in zip(hinode_dataset.data, hinode_dataset):
    imsave(os.path.join(evaluation_path, 'real_hinode_%s.jpg' % os.path.basename(f)), d[0])

fetcher = HMIContinuumFetcher(ds_path=data_path)
fetcher.fetchDates(hinode_dates)

hmi_dataset = HMIContinuumDataset(data_path)
hmi_dataset.addEditor(PaddingEditor((hmi_shape, hmi_shape)))
loader = DataLoader(hmi_dataset, batch_size=1, shuffle=False)

trainer = Trainer(1, 1, upsampling=2, norm='in_rs_aff', lambda_diversity=0)
trainer.cuda()
iteration = trainer.resume(base_path)
print('Loaded Iteration %d' % iteration)


with torch.no_grad():
    for f, hmi_img in zip(hmi_dataset.data, loader):
        hmi_img = hmi_img[0, 0].float().detach().numpy()
        hmi_patches = view_as_blocks(hmi_img, (patch_shape, patch_shape))
        hmi_patches = np.reshape(hmi_patches, (-1, patch_shape, patch_shape))
        hinode_patches = []
        for hmi_patch in tqdm(hmi_patches):
            hinode_patch = trainer.forwardAB(torch.tensor(hmi_patch).cuda().view((1, 1, patch_shape, patch_shape)))
            hinode_patches.append(hinode_patch[0, 0].detach().cpu().numpy())


        hmi_patches = np.array(hmi_patches).reshape((n_patches, n_patches, patch_shape, patch_shape))
        hmi_img = np.array(hmi_patches).transpose(0,2,1,3).reshape(-1,hmi_patches.shape[1]*hmi_patches.shape[3])

        hinode_patches = np.array(hinode_patches).reshape((n_patches, n_patches, hmi_shape, hmi_shape))
        hinode_img = np.array(hinode_patches).transpose(0,2,1,3).reshape(-1,hinode_patches.shape[1]*hinode_patches.shape[3])

        imsave(os.path.join(evaluation_path, 'hmi_%s.jpg' % os.path.basename(f)), hmi_img)
        imsave(os.path.join(evaluation_path,'hinode_%s.jpg' % os.path.basename(f)), hinode_img)
