import glob
import os

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from itipy.data.editor import RandomPatchEditor
from itipy.train.util import skip_invalid

import torch

from itipy.data.dataset import SDODataset, StorageDataset, SOHODataset
from itipy.evaluation.compute_fid import saveImage, calculate_fid_given_paths

if __name__ == '__main__':
    base_path = '/gpfs/gpfs0/robert.jarolim/iti/ablation'
    sdo_path = "/gpfs/gpfs0/robert.jarolim/data/iti/sdo"
    sdo_converted_path = '/gpfs/gpfs0/robert.jarolim/data/converted/sdo_2048'
    soho_path = "/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep"
    soho_converted_path = '/gpfs/gpfs0/robert.jarolim/data/converted/soho_1024'
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')


    sdo_valid = SDODataset(sdo_path, patch_shape=(1024, 1024), resolution=2048, months=[11, 12])
    sdo_valid = StorageDataset(sdo_valid, sdo_converted_path)

    soho_valid = SOHODataset(soho_path, resolution=1024, months=[11, 12])
    soho_valid = StorageDataset(soho_valid, soho_converted_path, ext_editors=[RandomPatchEditor((512, 512))])

    path_A = os.path.join(base_path, 'A')
    path_B = os.path.join(base_path, 'B')

    upsample = nn.UpsamplingBilinear2d(scale_factor=2)
    loader = DataLoader(soho_valid, batch_size=4, num_workers=4)
    i, j = 0, 0
    for batch_A in tqdm(skip_invalid(loader), total=len(loader)):
        batch_A = batch_A.float()
        batch_A = upsample(batch_A)
        for sample in batch_A.detach().cpu().numpy():
            j += 1
            saveImage(j, path_A, sample)
    loader = DataLoader(sdo_valid, batch_size=4, num_workers=4)
    i = 0
    for batch_B in tqdm(skip_invalid(loader), total=len(loader)):
        for sample in batch_B.detach().cpu().numpy():
            i += 1
            saveImage(i, path_B, sample)

    fid_A = [calculate_fid_given_paths((os.path.join(path_B, dir), os.path.join(path_A, dir)), 4, device, 2048)
             for dir in os.listdir(path_B)]

    model_paths = sorted(glob.glob(os.path.join(base_path, '**/generator_AB.pt'), recursive=True))

    for model_path in model_paths:
        evaluation_path = model_path.replace('generator_AB.pt', 'fid')
        path_AB = os.path.join(evaluation_path, 'AB')

        model = torch.load(model_path)

        model.to(device)
        model.eval()

        loader = DataLoader(soho_valid, batch_size=4, num_workers=4)
        i, j = 0, 0
        for batch_A in tqdm(skip_invalid(loader), total=len(loader)):
            batch_A = batch_A.to(device).float()
            with torch.no_grad():
                batch_AB = model.forward(batch_A)
            for sample in batch_AB.detach().cpu().numpy():
                i += 1
                saveImage(i, path_AB, sample)

        fid_AB = [
            calculate_fid_given_paths((os.path.join(path_B, dir), os.path.join(path_AB, dir)), 4, device, 2048)
            for dir in os.listdir(path_B)]

        with open(os.path.join(evaluation_path, "FID.txt"), "w") as text_file:
            text_file.writelines(['Samples A: %d; Samples B: %d\n' % (len(soho_valid), len(sdo_valid)),
                                  'AB: %s\n' % str(fid_AB), ])
