import glob
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from iti.data.editor import LoadMapEditor, NormalizeRadiusEditor, AIAPrepEditor, MapToDataEditor

base_path = '/gss/r.jarolim/data/ch_detection'
channels = [
    '131',
    '171',
    '193',
    '211',
    '304',
    '335',
    '94',
]

channel_files = [sorted(glob.glob(os.path.join(base_path, c, '*.fits'))) for c in channels]


def getMaxIntensity(f):
    s_map, _ = LoadMapEditor().call(f)
    s_map = NormalizeRadiusEditor(4096).call(s_map)
    s_map = AIAPrepEditor().call(s_map)
    data, _ = MapToDataEditor().call(s_map)
    return np.max(data)


for c, c_files in zip(channels, channel_files):
    with Pool(8) as p:
        maxs = [m for m in tqdm(p.imap_unordered(getMaxIntensity, c_files[::50]), total=len(c_files[::50]))]
    print(c, 'MAX:', np.mean(maxs) + 0.5 * np.std(maxs))
