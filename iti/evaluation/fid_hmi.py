import os

from iti.data.editor import RandomPatchEditor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import pandas
import torch

from iti.data.dataset import StorageDataset, HMIContinuumDataset, HinodeDataset
from iti.evaluation.compute_fid import computeFID

if __name__ == '__main__':
    df = pandas.read_csv('/gss/r.jarolim/data/hinode/file_list.csv', index_col=False, parse_dates=['date'])
    test_df = df[(df.date.dt.month == 12) | (df.date.dt.month == 11)]
    hinode_dataset = HinodeDataset(list(test_df.file))
    hinode_dataset = StorageDataset(hinode_dataset, '/gss/r.jarolim/data/converted/hinode_continuum',
                                    ext_editors=[RandomPatchEditor((640, 640))])

    # Init Dataset
    hmi_dataset = HMIContinuumDataset("/gss/r.jarolim/data/hmi_continuum/6173", months=[11, 12])
    hmi_dataset = StorageDataset(hmi_dataset, '/gss/r.jarolim/data/converted/hmi_continuum',
                                 ext_editors=[RandomPatchEditor((160, 160))])

    model = torch.load('/gss/r.jarolim/iti/hmi_hinode_v12/generator_AB.pt')

    evaluation_path = '/gss/r.jarolim/iti/hmi_hinode_v12/fid'
    fid_AB, fid_A = computeFID(evaluation_path, hmi_dataset, hinode_dataset, model, 1,
                               scale_factor=4)
    with open(os.path.join(evaluation_path, "FID.txt"), "w") as text_file:
        text_file.writelines(['Samples A: %d; Samples B: %d \n' % (len(hmi_dataset), len(hinode_dataset)),
                              'AB: %s\n' % str(fid_AB),
                              'A: %s\n' % str(fid_A),])
