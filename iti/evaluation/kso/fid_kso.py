import os


import torch

from iti.data.dataset import KSOFlatDataset, StorageDataset
from iti.evaluation.compute_fid import computeFID

if __name__ == '__main__':
    base_path = '/gpfs/gpfs0/robert.jarolim/iti/kso_quality_v1'
    model = torch.load(os.path.join(base_path, 'generator_AB.pt'))
    evaluation_path = os.path.join(base_path, 'fid')

    resolution = 1024
    low_path = "/gpfs/gpfs0/robert.jarolim/data/iti/kso_q2"
    low_converted_path = '/gpfs/gpfs0/robert.jarolim/data/converted/kso_q2_flat_%d' % resolution
    high_path = "/gpfs/gpfs0/robert.jarolim/data/iti/kso_q1"
    high_converted_path = '/gpfs/gpfs0/robert.jarolim/data/converted/kso_q1_flat_%d' % resolution

    q1_dataset = KSOFlatDataset(high_path, resolution, months=[11, 12])
    q1_dataset = StorageDataset(q1_dataset, high_converted_path)
    q2_dataset = KSOFlatDataset(low_path, resolution, months=[11, 12])
    q2_dataset = StorageDataset(q2_dataset, low_converted_path)


    fid_AB, fid_A = computeFID(evaluation_path, q2_dataset, q1_dataset, model)
    with open(os.path.join(evaluation_path, "FID.txt"), "w") as text_file:
        text_file.writelines(['Samples A: %d; Samples B: %d\n' % (len(q2_dataset), len(q1_dataset)),
                              'AB: %s\n' % str(fid_AB),
                              'A: %s\n' % str(fid_A),])
