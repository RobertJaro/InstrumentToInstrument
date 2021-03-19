import glob
import os

from astropy.io.fits import getheader, getdata
from matplotlib import pyplot as plt
from tqdm import tqdm

from sunpy.map import Map

import numpy as np

mins = []
maxs = []
means = []
for f in tqdm(sorted(glob.glob(os.path.join('/gss/r.jarolim/data/hinode/level1', '*.fits')))):
    data = getdata(f) / getheader(f)['EXPTIME']
    plt.imshow(data)
    plt.colorbar()
    plt.savefig('/gss/r.jarolim/data/hinode/imgs/%s.jpg' % os.path.basename(f), dpi=80)
    plt.close()
    mins.append(np.min(data))
    maxs.append(np.max(data))
    means.append(np.mean(data))

plt.plot(maxs)
plt.plot(means)
plt.plot(mins)

plt.savefig('/gss/r.jarolim/data/hinode/lightcurve.jpg')
plt.close()



maxs = []
for f in tqdm(sorted(glob.glob('/gss/r.jarolim/data/hinode/gband/*.fits'))):
    data = getdata(f) / getheader(f)['EXPTIME']
    maxs.append(np.max(data))

plt.hist(maxs)
plt.savefig('/gss/r.jarolim/data/hinode/gband_hist_max.jpg')
plt.close()