import torch

from iti.data.dataset import KSOFlatDataset, SDODataset, StorageDataset, SOHODataset
from iti.evaluation.compute_fid import computeFID

if __name__ == '__main__':
    sdo_valid = SDODataset("/gss/r.jarolim/data/ch_detection", patch_shape=(1024, 1024), resolution=2048,
                           months=[11, 12], limit=100)
    sdo_valid = StorageDataset(sdo_valid, '/gss/r.jarolim/data/converted/sdo_2048')

    soho_valid = SOHODataset("/gss/r.jarolim/data/soho_iti2021_prep", resolution=1024, months=[11, 12], limit=100)
    soho_valid = StorageDataset(soho_valid, '/gss/r.jarolim/data/converted/soho_1024')

    model = torch.load('/gss/r.jarolim/iti/soho_sdo_v24/generator_AB.pt')

    fid_AB, fid_A = computeFID('/gss/r.jarolim/iti/soho_sdo_v24/fid', soho_valid, sdo_valid, model, 2, scale_factor=2)
    print('FID AB: ', fid_AB)
    print('FID A: ', fid_A)