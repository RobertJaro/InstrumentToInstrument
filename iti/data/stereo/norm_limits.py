import glob
import os
import numpy as np
from astropy.io.fits import getdata, getheader
from matplotlib import pyplot as plt
from tqdm import tqdm

base_path = '/gss/r.jarolim/data/soho/valid'
channels = [
'eit_171',
'eit_195',
'eit_284',
'eit_304',
]

channel_files = [glob.glob(os.path.join(base_path, c, '*.fits')) for c in channels]

for c_files in channel_files:
    maxs = []
    exps = []
    for f in tqdm(c_files[::5]):
        data = getdata(f)
        maxs.append(np.max(data))
    print('MAX:', np.mean(maxs) + np.std(maxs))

