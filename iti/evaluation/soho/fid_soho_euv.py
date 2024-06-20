import os

from iti.data.editor import LambdaEditor



import torch

from iti.data.dataset import SDODataset, StorageDataset, SOHODataset
from iti.evaluation.compute_fid import computeFID

if __name__ == '__main__':
    base_path = '/gpfs/gpfs0/robert.jarolim/iti/soho_sdo_euv_v1'
    model = torch.load(os.path.join(base_path, 'generator_AB.pt'))
    evaluation_path = os.path.join(base_path, 'fid')

    sdo_path = "/gpfs/gpfs0/robert.jarolim/data/iti/sdo"
    sdo_converted_path = '/gpfs/gpfs0/robert.jarolim/data/converted/sdo_2048'
    soho_path = "/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep"
    soho_converted_path = '/gpfs/gpfs0/robert.jarolim/data/converted/soho_1024'

    channel_editor = LambdaEditor(lambda x, **kwargs: x[:-1])
    sdo_valid = SDODataset(sdo_path, patch_shape=(1024, 1024), resolution=2048,
                           months=[11, 12])
    sdo_valid = StorageDataset(sdo_valid, sdo_converted_path, ext_editors=[channel_editor])

    soho_valid = SOHODataset(soho_path, resolution=1024, months=[11, 12])
    soho_valid = StorageDataset(soho_valid, soho_converted_path, ext_editors=[channel_editor])

    fid_AB, fid_A = computeFID(evaluation_path, soho_valid, sdo_valid, model, 2, scale_factor=2)
    with open(os.path.join(evaluation_path, "FID.txt"), "w") as text_file:
        text_file.writelines(['Samples A: %d; Samples B: %d\n' % (len(soho_valid), len(sdo_valid)),
                              'AB: %s\n' % str(fid_AB),
                              'A: %s\n' % str(fid_A), ])
