import torch

from iti.data.dataset import KSOFlatDataset
from iti.evaluation.compute_fid import computeFID

if __name__ == '__main__':
    q1_dataset = KSOFlatDataset("/gss/r.jarolim/data/kso_synoptic", 1024)
    q2_dataset = KSOFlatDataset("/gss/r.jarolim/data/kso_general/quality2", 1024)
    model = torch.load('/gss/r.jarolim/iti/kso_quality_1024_v3/generator_AB.pt')

    fid_AB, fid_A = computeFID('/gss/r.jarolim/iti/kso_quality_1024_v3/fid', q2_dataset, q1_dataset, model)
    print('FID AB: ', fid_AB)
    print('FID A: ', fid_A)