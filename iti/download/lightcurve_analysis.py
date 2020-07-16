import glob
import os

from astropy.io.fits import getheader, getdata
from matplotlib import pyplot as plt
from tqdm import tqdm

import numpy as np

mins = []
maxs = []
means = []
for f in tqdm(sorted(glob.glob(os.path.join('/gss/r.jarolim/data/hinode_level1/level1', '*.fits')))):
    data = getdata(f)
    mins.append(np.min(data))
    maxs.append(np.max(data))
    means.append(np.mean(data))

plt.plot(maxs)
plt.plot(means)
plt.plot(mins)

plt.savefig('/gss/r.jarolim/data/hinode_level1/lightcurve.jpg')