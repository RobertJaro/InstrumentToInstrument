import datetime
import glob
import os
from warnings import simplefilter

import pandas
import pandas as pd
import torch
from dateutil.parser import parse
from itipy.train.util import skip_invalid
from sunpy.map import Map
from sunpy.visualization.colormaps import cm

from itipy.data.editor import soho_norms, sdo_norms, stereo_norms

from itipy.data.dataset import SOHODataset, STEREODataset, SDODataset, get_intersecting_files
from torch.utils.data import DataLoader
from tqdm import tqdm

from itipy.translate import SOHOToSDOEUV, SOHOToSDO

from itipy.translate import STEREOToSDO

from matplotlib import pyplot as plt

import numpy as np

# init
base_path = '/gpfs/gpfs0/robert.jarolim/iti/euv_comparison_v1'
os.makedirs(base_path, exist_ok=True)

translator_soho = SOHOToSDO(model_path='/gpfs/gpfs0/robert.jarolim/iti/soho_sdo_v4/generator_AB.pt')

# translate
basenames_soho = [[os.path.basename(f) for f in glob.glob('/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep/%s/*.fits' % wl)] for
                  wl in ['171', '195', '284', '304', 'mag']]
basenames_soho = set(basenames_soho[0]).intersection(*basenames_soho[1:])
basenames_soho = [b for b in basenames_soho if b not in ['2010-05-13T01:19:39.fits', '2010-05-12T19:19:38.fits', '2010-05-13T07:19:37.fits']]
basenames_sdo = [[os.path.basename(f) for f in glob.glob('/gpfs/gpfs0/robert.jarolim/data/iti/sdo_comparison/%s/*.fits' % wl)] for wl in
                 ['171', '193', '211', '304', '6173']]
basenames_sdo = set(basenames_sdo[0]).intersection(*basenames_sdo[1:])

dates_soho = sorted([parse(f.split('.')[0]) for f in basenames_soho])
dates_sdo = sorted([parse(f.split('.')[0]) for f in basenames_sdo])

closest_dates = [(date_soho, min(dates_sdo, key=lambda x: abs(x - date_soho))) for date_soho in dates_soho]
selected_dates = [(date_soho, date_sdo) for date_soho, date_sdo in closest_dates if
                  np.abs(date_soho - date_sdo) < datetime.timedelta(minutes=30) and date_sdo.year == 2010] # file name filter (below filter < 30 min)
selected_dates = selected_dates[::10]

basenames_soho = ['%s.fits' % date_soho.isoformat('T') for date_soho, date_sdo in selected_dates]
basenames_sdo = ['%s.fits' % date_sdo.isoformat('T') for date_soho, date_sdo in selected_dates]

soho_files = [['/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep/%s/%s' % (dir, basename) for basename in basenames_soho]
               for dir in ['171', '195', '284', '304', 'mag']]
soho_dataset = SOHODataset(soho_files)
soho_iterator = DataLoader(soho_dataset, batch_size=1, shuffle=False, num_workers=4)

sdo_files = [['/gpfs/gpfs0/robert.jarolim/data/iti/sdo_comparison/%s/%s' % (dir, basename) for basename in basenames_sdo]
              for dir in ['171', '193', '211', '304', '6173']]
sdo_dataset = SDODataset(sdo_files)
sdo_iterator = DataLoader(sdo_dataset, batch_size=1, shuffle=False, num_workers=4)

cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    'gray'
]
channel_mapping = {s:t for s,t in zip([171, 195, 284, 304], [171, 193, 211, 304])}
eit_calibration = {'171': [113.69278, 40.340622], '195': [60.60053, 31.752682], '284': [4.7249465, 3.9555929], '304': [64.73511, 26.619505]}
aia_calibration = {'171': [148.90274, 62.101795], '193': [146.01889, 71.47675], '211': [44.460854, 27.592617], '304': [46.21493, 18.522688]}

results = {wl: [] for wl in [171, 195, 284, 304]}

for soho_cube, sdo_cube in tqdm(skip_invalid(zip(soho_iterator, sdo_iterator)), total=len(selected_dates)):
    with torch.no_grad():
        iti_cube = translator_soho.generator(soho_cube.cuda())
        iti_cube = iti_cube[0].cpu().numpy()
    soho_cube = soho_cube[0].numpy()
    sdo_cube = sdo_cube[0].numpy()
    #
    for i, wl in enumerate([171, 195, 284, 304]):
        #
        original_mean = np.mean(soho_norms[wl].inverse((soho_cube[i] + 1) / 2))
        #
        sdo_norm = sdo_norms[channel_mapping[wl]]
        iti_mean = np.mean(sdo_norm.inverse((iti_cube[i] + 1) / 2))
        #
        eit_mean, eit_std = eit_calibration[str(wl)]
        aia_mean, aia_std = aia_calibration[str(channel_mapping[wl])]
        calibrated_mean = (np.array(original_mean) - eit_mean) * (aia_std / eit_std) + aia_mean
        #
        true_mean = np.mean(sdo_norm.inverse((sdo_cube[i] + 1) / 2))
        #
        results[wl] += [(original_mean, calibrated_mean, iti_mean, true_mean)]

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(soho_cube[0], vmin=-1, vmax=1, cmap=cm.sdoaia171)
axs[1].imshow(iti_cube[0], vmin=-1, vmax=1, cmap=cm.sdoaia171)
axs[2].imshow(sdo_cube[0], vmin=-1, vmax=1, cmap=cm.sdoaia171)
[ax.set_axis_off() for ax in axs]
plt.tight_layout(pad=0)
plt.savefig('/gpfs/gpfs0/robert.jarolim/iti/euv_comparison_v1/comparison.jpg')
plt.close()

with open(os.path.join(base_path, 'soho_evaluation.txt'), 'w') as f:
    for k, v in results.items():
        means = np.array(v)
        print(k, 'MAE', file=f)
        print('original', np.abs(means[:, -1] - means[:, 0]).mean(), file=f)
        print('calibrated', np.abs(means[:, -1] - means[:, 1]).mean(), file=f)
        print('iti', np.abs(means[:, -1] - means[:, 2]).mean(), file=f)
        print(k, 'CC', file=f)
        print('original', np.corrcoef(means[:, -1], means[:, 0])[0, 1], file=f)
        print('calibrated', np.corrcoef(means[:, -1], means[:, 1])[0, 1], file=f)
        print('iti', np.corrcoef(means[:, -1], means[:, 2])[0, 1], file=f)
    #
    print('Means', file=f)
    means = np.array(list(results.values()))
    print('original', np.abs(means[:, : , -1] - means[:, :, 0]).mean(), file=f)
    print('calibrated', np.abs(means[:, :, -1] - means[:, :, 1]).mean(), file=f)
    print('iti', np.abs(means[:, :, -1] - means[:, :, 2]).mean(), file=f)