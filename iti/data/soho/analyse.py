import glob

from astropy.io.fits import getheader, getdata
import numpy as np
from tqdm import tqdm

def evaluateMinMax(path):
    files = glob.glob(path)
    mins = []
    maxs = []
    for f in tqdm(files):
        data = getdata(f)
        if data.shape != (1024, 1024):
            print(data.shape)
            continue
        mins.append(np.min(data))
        maxs.append(np.max(data))
    print('min', np.mean(mins))
    print('max', np.mean(maxs))


evaluateMinMax('/gss/r.jarolim/data/soho/train/eit_171/*.fits')
evaluateMinMax('/gss/r.jarolim/data/soho/train/eit_195/*.fits')
evaluateMinMax('/gss/r.jarolim/data/soho/train/eit_284/*.fits')
evaluateMinMax('/gss/r.jarolim/data/soho/train/eit_304/*.fits')