import glob
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from iti.data.editor import LoadMapEditor, NormalizeRadiusEditor, AIAPrepEditor, MapToDataEditor

base_path = '/gss/r.jarolim/data/soho/valid'
channels = [
'eit_171',
'eit_195',
'eit_284',
'eit_304',
]

channel_files = [sorted(glob.glob(os.path.join(base_path, c, '*.fits'))) for c in channels]

def getMaxIntensity(f):
    s_map, _ = LoadMapEditor().call(f)
    s_map = NormalizeRadiusEditor(1024).call(s_map)
    data, _ = MapToDataEditor().call(s_map)
    return np.max(data)

for c, c_files in zip(channels, channel_files):
    with Pool(8) as p:
        maxs = [m for m in tqdm(p.imap_unordered(getMaxIntensity, c_files), total=len(c_files))]
    print(c, 'MAX:', np.mean(maxs) + 0.5 * np.std(maxs))
