import glob
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from dateutil.parser import parse
from tqdm import tqdm

import matplotlib.dates as mdates

from itipy.data.editor import LoadMapEditor, NormalizeRadiusEditor, MapToDataEditor, EITCheckEditor, RemoveOffLimbEditor, \
    AIAPrepEditor, SECCHIPrepEditor

stereo_path = '/gpfs/gpfs0/robert.jarolim/data/iti/stereo_iti2021_prep'

evaluation_path = '/gpfs/gpfs0/robert.jarolim/iti/euv_calibration'
soho_channels = ['195', '284']

secchi_files = [sorted(glob.glob(os.path.join(stereo_path, c, '*.fits'))) for c in soho_channels]


def getQSdata(f):
    s_map, _ = LoadMapEditor().call(f)
    s_map = SECCHIPrepEditor().call(s_map)
    s_map = NormalizeRadiusEditor(1024).call(s_map)
    s_map = RemoveOffLimbEditor(fill_value=np.nan).call(s_map)
    data, _ = MapToDataEditor().call(s_map)
    threshold = np.nanmedian(data) + np.nanstd(data)
    data[data > threshold] = np.nan
    return data


secchi_means = {}
for c, c_files in zip(soho_channels, secchi_files):
    c_files = c_files[::len(c_files) // 100]
    dates = [parse(os.path.basename(f).replace('.fits', '')) for f in c_files]
    with Pool(4) as p:
        means = [np.nanmean(m) for m in tqdm(p.imap(getQSdata, c_files), total=len(c_files))]
    secchi_means[c] = (dates, means)


for c, (secchi_dates, y) in secchi_means.items():
    x = mdates.date2num(secchi_dates)
    fit_params = np.polyfit(x, y, 1)
    fit = np.poly1d(fit_params)
    d0 = fit(x[0])
    print(c, fit_params / d0)
    plt.plot(x, np.array(y) / fit(x), label='Quiet-Sun Mean')
    # plt.plot(x, fit(x), label='Fit', linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(evaluation_path, '%s_mean.jpg' % c))
    plt.close()

