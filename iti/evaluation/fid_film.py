import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch

from iti.data.dataset import SDODataset, StorageDataset, SOHODataset, KSOFlatDataset, KSOFilmDataset
from iti.evaluation.compute_fid import computeFID

if __name__ == '__main__':
    resolution = 512
    ccd_dataset = KSOFlatDataset("/gss/r.jarolim/data/kso_synoptic", resolution, months=[11, 12])
    ccd_dataset = StorageDataset(ccd_dataset, '/gss/r.jarolim/data/converted/iti/kso_synoptic_q1_flat_%d' % resolution)

    film_dataset = KSOFilmDataset("/gss/r.jarolim/data/filtered_kso_plate", resolution, months=[11, 12])
    film_dataset = StorageDataset(film_dataset, '/gss/r.jarolim/data/converted/iti/kso_film_%d' % resolution)

    model = torch.load('/gss/r.jarolim/iti/film_v8/generator_AB.pt')

    evaluation_path = '/gss/r.jarolim/iti/film_v8/fid'
    fid_AB, fid_A = computeFID(evaluation_path, film_dataset, ccd_dataset, model, 2)
    with open(os.path.join(evaluation_path, "FID.txt"), "w") as text_file:
        text_file.writelines(['Samples A: %d; Samples B: %d\n' % (len(film_dataset), len(ccd_dataset)),
                              'AB: %s\n' % str(fid_AB),
                              'A: %s\n' % str(fid_A),])
