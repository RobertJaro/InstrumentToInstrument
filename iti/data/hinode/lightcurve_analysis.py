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
maxs_exp = []
exp_times = []
for f in tqdm(sorted(glob.glob(os.path.join('/gss/r.jarolim/data/hinode/level1', '*.fits')))):
    data = getdata(f)
    exp_time = getheader(f)['EXPTIME']
    exp_times.append(exp_time)
    maxs_exp.append(np.max(data) / exp_time)
    maxs.append(np.max(data))

plt.subplot(311)
plt.hist(maxs, bins=100)
plt.subplot(312)
plt.hist(maxs_exp, bins=100)
plt.subplot(313)
plt.hist(exp_times, bins=100)
plt.savefig('/gss/r.jarolim/data/hinode/continuum_hist.jpg')
plt.close()