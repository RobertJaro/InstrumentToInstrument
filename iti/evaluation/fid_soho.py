import os

from iti.data.editor import RandomPatchEditor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch

from iti.data.dataset import SDODataset, StorageDataset, SOHODataset
from iti.evaluation.compute_fid import computeFID

if __name__ == '__main__':
    sdo_valid = SDODataset("/gss/r.jarolim/data/ch_detection", patch_shape=(1024, 1024), resolution=2048,
                           months=[11, 12])
    sdo_valid = StorageDataset(sdo_valid, '/gss/r.jarolim/data/converted/sdo_2048')

    soho_valid = SOHODataset("/gss/r.jarolim/data/soho_iti2021_prep", resolution=1024, months=[11, 12])
    soho_valid = StorageDataset(soho_valid, '/gss/r.jarolim/data/converted/soho_1024', ext_editors=[RandomPatchEditor((512, 512))])

    model = torch.load('/gss/r.jarolim/iti/soho_sdo_v25/generator_AB.pt')

    evaluation_path = '/gss/r.jarolim/iti/soho_sdo_v25/fid'
    fid_AB, fid_A = computeFID(evaluation_path, soho_valid, sdo_valid, model, 2, scale_factor=2)
    with open(os.path.join(evaluation_path, "FID.txt"), "w") as text_file:
        text_file.writelines(['Samples A: %d; Samples B: %d\n' % (len(soho_valid), len(sdo_valid)),
                              'AB: %s\n' % str(fid_AB),
                              'A: %s\n' % str(fid_A),])
