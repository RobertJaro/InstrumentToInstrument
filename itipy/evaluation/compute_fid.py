import os
import shutil

import matplotlib.pyplot as plt
import torch
from pytorch_fid.fid_score import compute_statistics_of_path, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from itipy.train.model import GeneratorAB
from itipy.trainer import skip_invalid

def calculate_fid_given_paths(paths, batch_size, device, dims):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

    model = InceptionV3([block_idx], normalize_input=False, resize_input=False).to(device)

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

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
    for batch_A in tqdm(skip_invalid(loader), total=len(loader)):
        batch_A = batch_A.to(device).float()
        with torch.no_grad():
            batch_AB = model.forward(batch_A)
        for sample in batch_AB.detach().cpu().numpy():
            i += 1
            saveImage(i, path_AB, sample)
        batch_A = upsample(batch_A)
        for sample in batch_A.detach().cpu().numpy():
            j += 1
            saveImage(j, path_A, sample)

    loader = DataLoader(dataset_B, batch_size=batch_size, num_workers=4)
    i = 0
    for batch_B in tqdm(skip_invalid(loader), total=len(loader)):
        for sample in batch_B.detach().cpu().numpy():
            i += 1
            saveImage(i, path_B, sample)

    fid_AB = [calculate_fid_given_paths((os.path.join(path_B, dir), os.path.join(path_AB, dir)), batch_size, device, 2048) for dir in os.listdir(path_B)]
    fid_A = [calculate_fid_given_paths((os.path.join(path_B, dir), os.path.join(path_A, dir)), batch_size, device, 2048) for dir in os.listdir(path_B)]
    shutil.rmtree(path_A), shutil.rmtree(path_B), shutil.rmtree(path_AB) # clean up
    return fid_AB, fid_A


def saveImage(img_id, path, sample):
    for c in range(sample.shape[0]):
        os.makedirs(os.path.join(path, 'channel%d' % c), exist_ok=True)
        file_path = os.path.join(path, 'channel%d' % c, '%d.jpg' % img_id)
        plt.imsave(file_path, sample[c], vmin=-1, vmax=1, cmap='gray')


