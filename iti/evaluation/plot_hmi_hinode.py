import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from skimage.util import view_as_blocks
from torch.utils.data import DataLoader
from tqdm import tqdm

from iti.data.dataset import HMIContinuumDataset
from iti.data.editor import PaddingEditor
from iti.train.trainer import Trainer

hmi_shape = 4096
patch_shape = 1024
n_patches = hmi_shape // patch_shape
base_path = '/gss/r.jarolim/iti/hmi_hinode_v10'
os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)

hmi_dataset = HMIContinuumDataset("/gss/r.jarolim/data/hmi_continuum/6173")
hmi_dataset.addEditor(PaddingEditor((hmi_shape, hmi_shape)))
loader = DataLoader(hmi_dataset, batch_size=1, shuffle=True)
iter = loader.__iter__()

trainer = Trainer(1, 1, upsampling=2, norm='in_rs_aff', lambda_diversity=0)
trainer.cuda()
iteration = trainer.resume(base_path, epoch=140000)


with torch.no_grad():
    hmi_img = next(iter).float()
    hmi_img = hmi_img[0, 0].detach().numpy()
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

imsave(os.path.join(base_path, 'evaluation/full_disc_hmi_%d.jpg' % iteration), hmi_img)
imsave(os.path.join(base_path,'evaluation/full_disc_hinode_%d.jpg' % iteration), hinode_img)
