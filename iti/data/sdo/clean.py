import glob
import os
from multiprocessing import Pool

import numpy as np
from astropy.io.fits import getheader
from tqdm import tqdm

path = '/gpfs/gpfs0/robert.jarolim/data/iti/sdo'
wls = [171, 193, 211, 304]

def check(f):
    header = getheader(f, 1)
    if header['QUALITY'] != 0:
        print(f, header['QUALITY'])
        fn = os.path.basename(f)
        [os.remove(os.path.join(path, str(wl), fn)) for wl in wls + [6173] if
         os.path.exists(os.path.join(path, str(wl), fn))]

for scan_wl in wls:
    files = glob.glob(os.path.join(path, str(scan_wl),'*.fits'))
    with Pool(8) as p:
        [None for _ in tqdm(p.imap_unordered(check, files), total=len(files))]


def convert(f):
    try:
        d = np.load(f)['arr_0']
    except:
        os.remove(f)
        return
    if np.any(np.isnan(d)):
        os.remove(f)
        print('NAN encountered')
        return
    np.save(f.replace('npz', 'npy'), d)
    os.remove(f)

with Pool(8) as p:
    files = glob.glob('/gpfs/gpfs0/robert.jarolim/data/converted/sdo_2048/*.npz')
    [None for _ in tqdm(p.imap_unordered(convert, files), total=len(files))]