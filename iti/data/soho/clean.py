import glob
import os

from astropy.io.fits import getheader, getdata
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

def clean(path):
    files = glob.glob(path)
    for f in tqdm(files):
        data = getdata(f)
        header = getheader(f)
        if data.shape != (1024, 1024) or 'N_MISSING_BLOCKS =    0' not in header['COMMENT'][-1]:
            print(data.shape)
            print(header['COMMENT'][-1])
            #os.remove(f)
            continue
        #print('valid')

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
            #os.remove(f)
            continue
        #print('valid')

#clean('/gss/r.jarolim/data/soho/train/eit_171/*.fits')
#clean('/gss/r.jarolim/data/soho/train/eit_195/*.fits')
#clean('/gss/r.jarolim/data/soho/train/eit_284/*.fits')
#clean('/gss/r.jarolim/data/soho/train/eit_304/*.fits')
#clean_mdi('/gss/r.jarolim/data/soho/train/mdi_mag/*.fits')

light_curve = getLightCurve('/gss/r.jarolim/data/soho/train/mdi_mag/*.fits')
plt.scatter(range(len(light_curve)), light_curve[:, 0].astype(np.float))
plt.savefig('/gss/r.jarolim/data/converted/iti_samples/soho_lc_mag.jpg')
plt.close()

invalid_files = light_curve[:, 1][np.abs(light_curve[:, 0].astype(np.float)) > 10]

invalid_file_path = '/gss/r.jarolim/data/soho/train/mdi_mag/2003-01-08T01:19.fits'
plt.imshow(getdata(invalid_file_path), cmap='gray', vmin=-100, vmax=100)
print(np.nanmean(getdata(invalid_file_path)))
plt.savefig('/gss/r.jarolim/data/converted/iti_samples/invalid_mag.jpg')
plt.close()

# lists = [glob.glob('/gss/r.jarolim/data/soho_iti2021_prep/171/*.fits'),
#          glob.glob('/gss/r.jarolim/data/soho_iti2021_prep/195/*.fits'),
#          glob.glob('/gss/r.jarolim/data/soho_iti2021_prep/284/*.fits'),
#          glob.glob('/gss/r.jarolim/data/soho_iti2021_prep/304/*.fits'),
#          glob.glob('/gss/r.jarolim/data/soho_iti2021_prep/mag/*.fits'),]
# joined_files = set.intersection(*map(set,[[os.path.basename(f) for f in files] for files in lists]))
#
#
#
# [len([os.remove(f) for f in files if os.path.basename(f) not in joined_files]) for files in lists]