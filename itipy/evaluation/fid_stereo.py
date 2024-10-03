import os

from itipy.data.editor import RandomPatchEditor, SliceEditor



import torch

from itipy.data.dataset import SDODataset, StorageDataset, STEREODataset
from itipy.evaluation.compute_fid import computeFID

if __name__ == '__main__':
    base_path = '/gpfs/gpfs0/robert.jarolim/iti/stereo_to_sdo_v1'
    model = torch.load(os.path.join(base_path, 'generator_AB.pt'))
    evaluation_path = os.path.join(base_path, 'fid')

    stereo_path = "/gpfs/gpfs0/robert.jarolim/data/iti/stereo_iti2021_prep"
    stereo_converted_path = '/gpfs/gpfs0/robert.jarolim/data/converted/stereo_1024_calibrated'
    sdo_path = "/gpfs/gpfs0/robert.jarolim/data/iti/sdo"
    sdo_converted_path = '/gpfs/gpfs0/robert.jarolim/data/converted/sdo_4096'

    sdo_valid = StorageDataset(
        SDODataset(sdo_path, resolution=4096, patch_shape=(1024, 1024), months=[11, 12]),
        sdo_converted_path, ext_editors=[SliceEditor(0, -1)])
    stereo_valid = StorageDataset(
        STEREODataset(stereo_path, patch_shape=(1024, 1024), months=[11, 12]),
        stereo_converted_path, ext_editors=[RandomPatchEditor((256, 256))])

    fid_AB, fid_A = computeFID(evaluation_path, stereo_valid, sdo_valid, model, 1, scale_factor=4)
    with open(os.path.join(evaluation_path, "FID.txt"), "w") as text_file:
        text_file.writelines(['Samples A: %d; Samples B: %d\n' % (len(stereo_valid), len(sdo_valid)),
                              'AB: %s\n' % str(fid_AB),
                              'A: %s\n' % str(fid_A), ])
