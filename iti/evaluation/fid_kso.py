import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch

from iti.data.dataset import KSOFlatDataset, StorageDataset
from iti.evaluation.compute_fid import computeFID

if __name__ == '__main__':
    resolution = 1024
    q1_dataset = KSOFlatDataset("/gss/r.jarolim/data/kso_synoptic", resolution, months=[11, 12])
    q1_dataset = StorageDataset(q1_dataset, '/gss/r.jarolim/data/converted/iti/kso_synoptic_q1_flat_%d' % resolution)

    q2_dataset = KSOFlatDataset('/gss/r.jarolim/data/kso_general/quality2', resolution, months=[11, 12])
    q2_dataset = StorageDataset(q2_dataset, '/gss/r.jarolim/data/converted/iti/kso_q2_flat_%d' % resolution)
    model = torch.load('/gss/r.jarolim/iti/kso_quality_1024_v7/generator_AB.pt')

    evaluation_path = '/gss/r.jarolim/iti/kso_quality_1024_v7/fid'
    fid_AB, fid_A = computeFID(evaluation_path, q2_dataset, q1_dataset, model)
    with open(os.path.join(evaluation_path, "FID.txt"), "w") as text_file:
        text_file.writelines(['Samples A: %d; Samples B: %d\n' % (len(q2_dataset), len(q1_dataset)),
                              'AB: %s\n' % str(fid_AB),
                              'A: %s\n' % str(fid_A),])
