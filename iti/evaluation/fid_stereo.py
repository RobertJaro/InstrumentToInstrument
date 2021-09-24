import os

from iti.data.editor import RandomPatchEditor, SliceEditor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch

from iti.data.dataset import SDODataset, StorageDataset, STEREODataset
from iti.evaluation.compute_fid import computeFID

if __name__ == '__main__':
    sdo_valid = StorageDataset(
        SDODataset("/gss/r.jarolim/data/sdo/valid", resolution=4096, patch_shape=(1024, 1024), months=[11, 12]),
        '/gss/r.jarolim/data/converted/sdo_4096', ext_editors=[SliceEditor(0, -1)])
    stereo_valid = StorageDataset(
        STEREODataset("/gss/r.jarolim/data/stereo_iti2021_prep", patch_shape=(1024, 1024), months=[11, 12]),
        '/gss/r.jarolim/data/converted/stereo_1024', ext_editors=[RandomPatchEditor((256, 256))])

    model = torch.load('/gss/r.jarolim/iti/stereo_v7/generator_AB.pt')

    evaluation_path = '/gss/r.jarolim/iti/stereo_v7/fid'
    fid_AB, fid_A = computeFID(evaluation_path, stereo_valid, sdo_valid, model, 1, scale_factor=4)
    with open(os.path.join(evaluation_path, "FID.txt"), "w") as text_file:
        text_file.writelines(['Samples A: %d; Samples B: %d\n' % (len(stereo_valid), len(sdo_valid)),
                              'AB: %s\n' % str(fid_AB),
                              'A: %s\n' % str(fid_A), ])
