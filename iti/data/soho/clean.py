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
        if data.shape != (1024, 1024) or 'N_MISSING_BLOCKS =    0' not in header['COMMENT'][-1]:
            print(data.shape)
            print(header['COMMENT'][-1])
            #os.remove(f)
            continue
        print('valid')

clean('/gss/r.jarolim/data/soho/valid/eit_171/*.fits')
clean('/gss/r.jarolim/data/soho/valid/eit_195/*.fits')
clean('/gss/r.jarolim/data/soho/valid/eit_284/*.fits')
clean('/gss/r.jarolim/data/soho/valid/eit_304/*.fits')

lists = [glob.glob('/gss/r.jarolim/data/soho/valid/eit_171/*.fits'),
         glob.glob('/gss/r.jarolim/data/soho/valid/eit_195/*.fits'),
         glob.glob('/gss/r.jarolim/data/soho/valid/eit_284/*.fits'),
         glob.glob('/gss/r.jarolim/data/soho/valid/eit_304/*.fits'),
         glob.glob('/gss/r.jarolim/data/soho/valid/mdi_mag/*.fits'),]
joined_files = set.intersection(*map(set,[[os.path.basename(f) for f in files] for files in lists]))

#[len([os.remove(f) for f in files if os.path.basename(f) not in joined_files]) for files in lists]