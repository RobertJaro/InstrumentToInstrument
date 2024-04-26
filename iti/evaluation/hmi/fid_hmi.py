import os

import pandas as pd

from iti.data.editor import RandomPatchEditor

import torch

from iti.data.dataset import StorageDataset, HMIContinuumDataset, HinodeDataset
from iti.evaluation.compute_fid import computeFID

if __name__ == '__main__':
    base_path = '/gpfs/gpfs0/robert.jarolim/iti/hmi_hinode_v4'
    model = torch.load(os.path.join(base_path, 'generator_AB.pt'))
    evaluation_path = os.path.join(base_path, 'fid')

    hmi_path = '/gpfs/gpfs0/robert.jarolim/data/iti/hmi_continuum'
    hmi_converted_path = '/gpfs/gpfs0/robert.jarolim/data/converted/hmi_continuum'
    hinode_converted_path = '/gpfs/gpfs0/robert.jarolim/data/converted/hinode_continuum'
    hinode_file_list = '/gpfs/gpfs0/robert.jarolim/data/iti/hinode_file_list.csv'

    df = pd.read_csv(hinode_file_list, index_col=False, parse_dates=['date'])
    test_df = df[df.date.dt.month.isin([11, 12])]
    hinode_dataset = HinodeDataset(list(test_df.file))
    hinode_dataset = StorageDataset(hinode_dataset, hinode_converted_path,
                                    ext_editors=[RandomPatchEditor((640, 640))])

    hmi_dataset = HMIContinuumDataset(hmi_path, months=[11, 12])
    hmi_dataset = StorageDataset(hmi_dataset, hmi_converted_path, ext_editors=[RandomPatchEditor((160, 160))])

    fid_AB, fid_A = computeFID(evaluation_path, hmi_dataset, hinode_dataset, model, 1,
                               scale_factor=4)
    with open(os.path.join(evaluation_path, "FID.txt"), "w") as text_file:
        text_file.writelines(['Samples A: %d; Samples B: %d \n' % (len(hmi_dataset), len(hinode_dataset)),
                              'AB: %s\n' % str(fid_AB),
                              'A: %s\n' % str(fid_A),])
