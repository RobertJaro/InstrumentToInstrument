import glob
from multiprocessing import Pool
from warnings import simplefilter

import numpy as np
import pandas as pd
from sunpy.map import Map, all_coordinates_from_map
from tqdm import tqdm

hinode_files = sorted(glob.glob('/gss/r.jarolim/data/hinode/level1/*.fits'))


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


with Pool(8) as p:
    file_list = [d for d in tqdm(p.imap_unordered(classify, hinode_files), total=len(hinode_files))]

pd.DataFrame(data=file_list, columns=['file', 'classification', 'date']).to_csv(
    '/gss/r.jarolim/data/hinode/file_list.csv', index=False)
