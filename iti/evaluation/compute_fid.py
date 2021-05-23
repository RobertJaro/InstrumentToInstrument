import os

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
from skimage.io import imsave
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from iti.data.dataset import KSOFlatDataset
from iti.train.model import GeneratorAB
from iti.train.trainer import Trainer


def computeFID(base_path, dataset_A, dataset_B, model:GeneratorAB, batch_size=4, scale_factor=1):
    path_A = os.path.join(base_path, 'A')
    path_B = os.path.join(base_path, 'B')
    path_AB = os.path.join(base_path, 'AB')

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    model.to(device)
    model.eval()
    upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    loader = DataLoader(dataset_A, batch_size=batch_size, num_workers=4)
    i, j = 0, 0
    for batch_A in tqdm(loader):
        batch_A = batch_A.to(device)
        with torch.no_grad():
            batch_AB = model.forward(batch_A)
        for sample in batch_AB.detach().cpu().numpy():
            i += 1
            saveBatch(i, path_AB, sample)
        batch_A = upsample(batch_A)
        for sample in batch_A.detach().cpu().numpy():
            j += 1
            saveBatch(j, path_A, sample)

    loader = DataLoader(dataset_B, batch_size=batch_size, num_workers=4)
    i = 0
    for batch_B in tqdm(loader):
        for sample in batch_B.detach().cpu().numpy():
            i += 1
            saveBatch(i, path_B, sample)

    fid_AB = [calculate_fid_given_paths((os.path.join(path_B, dir), os.path.join(path_AB, dir)), batch_size, device, 2048) for dir in os.listdir(path_B)]
    fid_A = [calculate_fid_given_paths((os.path.join(path_B, dir), os.path.join(path_A, dir)), batch_size, device, 2048) for dir in os.listdir(path_B)]
    return fid_AB, fid_A


def saveBatch(img_id, path, sample):
    for c in range(sample.shape[0]):
        os.makedirs(os.path.join(path, 'channel%d' % c), exist_ok=True)
        imsave(os.path.join(path, 'channel%d' % c, '%d.jpg' % img_id), sample[c], vmin=-1, vmax=1)


