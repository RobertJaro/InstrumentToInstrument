import glob
import os

from astropy.io.fits import getheader, getdata
import numpy as np
from tqdm import tqdm

def clean(path):
    files = glob.glob(path)
    mins = []
    maxs = []
    for f in tqdm(files):
        data = getdata(f)
        header = getheader(f)
        if data.shape != (1024, 1024) or header['NMISSING'] != 0:
            print(data.shape)
            print(header['NMISSING'])
            #os.remove(f)
            continue
        #print('valid')
        mins.append(header['DATAMIN'])
        maxs.append(header['DATAMAX'])
    print(np.median(mins), np.median(maxs))

ds_path = '/gss/r.jarolim/data/stereo_prep/valid'

clean(os.path.join(ds_path, 'secchi_171/*.fits'))
clean(os.path.join(ds_path, 'secchi_195/*.fits'))
clean(os.path.join(ds_path, 'secchi_284/*.fits'))
clean(os.path.join(ds_path, 'secchi_304/*.fits'))

lists = [glob.glob(os.path.join(ds_path, 'secchi_171/*.fits')),
         glob.glob(os.path.join(ds_path, 'secchi_195/*.fits')),
         glob.glob(os.path.join(ds_path, 'secchi_284/*.fits')),
         glob.glob(os.path.join(ds_path, 'secchi_304/*.fits')),]
joined_files = set.intersection(*map(set,[[os.path.basename(f) for f in files] for files in lists]))

#[len([os.remove(f) for f in files if os.path.basename(f) not in joined_files]) for files in lists]