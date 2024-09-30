import argparse
import glob
import os
from multiprocessing import Pool
from warnings import simplefilter

import numpy as np
import pandas as pd
from sunpy.map import Map, all_coordinates_from_map
from tqdm import tqdm


def classify(file):
    simplefilter('ignore')
    s_map = Map(file)
    hpc_coords = all_coordinates_from_map(s_map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
    if np.any(r > 1):
        return (file, 'limb', s_map.date)
    if np.std(s_map.data / s_map.meta['EXPTIME']) < 2500:
        return (file, 'quiet', s_map.date)
    return (file, 'feature', s_map.date)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coarse classification of Hinode files for optimized data sampling')
    parser.add_argument('--hinode_dir', type=str, help='path to the Hinode data.')
    parser.add_argument('--n_workers', type=str, help='number of parallel threads.', required=False, default=4)
    parser.add_argument('--result_path', type=str, help='path to print the result csv.')
    args = parser.parse_args()

    hinode_files = sorted(glob.glob(os.path.join(args.hinode_dir, '*.fits')))
    with Pool(args.n_workers) as p:
        file_list = [d for d in tqdm(p.imap_unordered(classify, hinode_files), total=len(hinode_files))]

    pd.DataFrame(data=file_list, columns=['file', 'classification', 'date']).to_csv(args.result_path, index=False)
