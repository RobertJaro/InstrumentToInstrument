import glob

import numpy as np
from astropy.io.fits import getheader, getdata
from sunpy.io.fits import get_header
from tqdm import tqdm


def clean(path, wl):
    files = glob.glob(path)
    for f in tqdm(files):
        data = getdata(f)
        header = getheader(f)
        if data.shape != (1024, 1024) or 'N_MISSING_BLOCKS =    0' not in header['COMMENT'][-1] or get_header(f)[0]['WAVELNTH'] != wl:
            print('invalid', f)
            print(data.shape)
            print(header['COMMENT'][-1])
            # os.remove(f)
            continue
        # print('valid')


def getLightCurve(path):
    sums = []
    files = sorted(glob.glob(path))
    for f in tqdm(files):
        data = getdata(f)
        sums += [(np.nanmean(data), f)]
    return np.array(sums)


def clean_mdi(path):
    files = glob.glob(path)
    for f in tqdm(files):
        data = getdata(f)
        header = getheader(f)
        if data.shape != (1024, 1024) or header['DPC_OBSR'] == 'FD_Continuum' or header['DPC_OBSR'] == 'FD_Doppler':
            print(data.shape)
            print(header['DPC_OBSR'], f)
            # os.remove(f)
            continue
        # print('valid')


clean('/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep/171/*.fits', 171)
