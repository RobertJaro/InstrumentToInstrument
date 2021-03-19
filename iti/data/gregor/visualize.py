import glob
import os
from warnings import simplefilter

from astropy.io import fits
from sunpy.map import Map

from matplotlib import pyplot as plt

import numpy as np

# scale
scale = 70 / 1280

maxs = []
for f in sorted(glob.glob('/gss/r.jarolim/data/gregor/*.fts')):
    simplefilter('ignore')
    hdul = fits.open(f)
    #
    if 'wavelnth' not in hdul[0].header:
        print('ERROR')
        print(hdul[0].header)
        print(hdul[1].header)
        continue
    if hdul[0].header['wavelnth'] == 430.7:
        index = 0
    elif hdul[1].header['wavelnth'] == 430.7:
        index = 1
    else:
        print('ERROR')
        print(hdul[0].header)
        print(hdul[1].header)
        continue
    #
    primary_header = hdul[0].header
    primary_header['cunit1'] = 'arcsec'
    primary_header['cunit2'] = 'arcsec'
    primary_header['cdelt1'] = scale
    primary_header['cdelt2'] = scale
    #
    g_band = hdul[index::2]
    g_band = sorted(g_band, key=lambda hdu: hdu.header['TIMEOFFS'])
    #
    print(hdul[0].header['DATE'], hdul[0].header['wavelnth'], hdul[1].header['wavelnth'])
    gregor_maps = [Map(hdu.data, primary_header) for hdu in g_band]
    #
    fig, axs = plt.subplots(10, 10, figsize=(60, 60))
    for ax, s_map in zip(np.ravel(axs), gregor_maps):
        s_map.plot(axes=ax, cmap='gray', vmin=0, vmax=1.8)
        maxs += [s_map.meta['DATAMAX']]
    #
    plt.tight_layout()
    plt.savefig('/gss/r.jarolim/data/converted/gregor_gband_img/%s.jpg' % os.path.basename(f), dpi=80)
    plt.close()

plt.hist(maxs)
plt.savefig('/gss/r.jarolim/data/converted/gregor_gband_img/hist.jpg')
plt.close()