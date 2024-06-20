import argparse
import os

import torch

from iti.data.dataset import KSOFlatDataset, KSOFilmDataset
from iti.evaluation.compute_fid import computeFID

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute FID for KSO film dataset')
    parser.add_argument('--out_path', type=str, help='Path to the output directory')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the dataset')
    parser.add_argument('--film_path', type=str, help='Path to the film dataset')
    parser.add_argument('--ccd_path', type=str, help='Path to the CCD dataset')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--months', type=int, nargs='+', help='Months to include in the dataset', default=[11, 12])

    args = parser.parse_args()

    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    ccd_dataset = KSOFlatDataset(args.film_path, args.resolution, months=args.months)
    film_dataset = KSOFilmDataset("/gss/r.jarolim/data/filtered_kso_plate", args.resolution, months=args.months)

    model = torch.load(args.model_path)

    fid_AB, fid_A = computeFID(out_path, film_dataset, ccd_dataset, model, 2)
    with open(os.path.join(out_path, "FID.txt"), "w") as text_file:
        text_file.writelines(['Samples A: %d; Samples B: %d\n' % (len(film_dataset), len(ccd_dataset)),
                              'AB: %s\n' % str(fid_AB),
                              'A: %s\n' % str(fid_A), ])
