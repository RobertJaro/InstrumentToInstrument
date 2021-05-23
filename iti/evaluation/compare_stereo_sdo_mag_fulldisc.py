import glob
import os
from datetime import timedelta, datetime

from dateutil.parser import parse
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from iti.prediction.translate import STEREOToSDOMagnetogram

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from matplotlib import pyplot as plt

from iti.data.dataset import SOHODataset

import numpy as np

stereo_shape = 256
base_path = "/gss/r.jarolim/iti/stereo_mag_v10"
prediction_path = os.path.join(base_path, 'evaluation')
os.makedirs(prediction_path, exist_ok=True)

soho_basenames = np.array([os.path.basename(f) for f in sorted(glob.glob('/gss/r.jarolim/data/soho_iti2021_prep/171/*.fits'))])
soho_dates = np.array([parse(bn.split('.')[0]) for bn in soho_basenames])
stereo_basenames = np.array(
    [os.path.basename(f) for f in sorted(glob.glob('/gss/r.jarolim/data/stereo_iti2021_prep/171/*.fits'))])
stereo_dates = np.array([parse(bn.split('.')[0]) for bn in stereo_basenames])

min_diff = np.array([np.min(np.abs(stereo_dates - soho_date)) for soho_date in soho_dates])
# filter time diff
cond = (min_diff < timedelta(hours=2)) & (soho_dates < datetime(2008, 1, 1))
soho_dates = soho_dates[cond]
soho_basenames = soho_basenames[cond]
# select corresponding stereo files
stereo_basenames = stereo_basenames[[np.argmin(np.abs(stereo_dates - soho_date)) for soho_date in soho_dates]]

soho_dataset = SOHODataset("/gss/r.jarolim/data/soho_iti2021_prep", basenames=soho_basenames, resolution=1024)
translator = STEREOToSDOMagnetogram(model_path=os.path.join(base_path, 'generator_AB.pt'))

result = translator.translate("/gss/r.jarolim/data/stereo_iti2021_prep", basenames=stereo_basenames)

sdo_cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    plt.get_cmap('gray')
]

mean_mag_iti = []
mean_mag_soho = []
dates = []

for date, (iti_maps, stereo_img, iti_img), soho_img in tqdm(zip(soho_dates, result, soho_dataset), total=len(stereo_basenames)):
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    if abs(iti_maps[0].meta['hgln_obs']) > 5:
        print(iti_maps[0].meta['hgln_obs'])
        continue
    for c in range(4):
        axs[0, c].imshow(stereo_img[c], cmap=sdo_cmaps[c])

    for c in range(4):
        axs[1, c].imshow(iti_img[c], cmap=sdo_cmaps[c], vmin=-1, vmax=1)

    axs[1, 4].imshow(iti_img[4], cmap=sdo_cmaps[4], vmin=-1, vmax=1)

    for c in range(4):
        axs[2, c].imshow(soho_img[c], cmap=sdo_cmaps[c], vmin=-1, vmax=1)

    axs[2, 4].imshow(np.abs(soho_img[4]), cmap=sdo_cmaps[4], vmin=0, vmax=1)

    [ax.set_axis_off() for ax in np.ravel(axs)]

    plt.tight_layout(0.1)
    plt.savefig(os.path.join(base_path, 'evaluation/%s.jpg' % date.isoformat('T')), dpi=300)
    plt.close()

    dates.append(date)
    mean_mag_iti.append(((iti_img[4] + 1) / 2).mean() * 1000)
    mean_mag_soho.append(np.abs(soho_img[4]).mean() * 1000)

    plt.figure(figsize=(8, 4))
    plt.plot(dates, mean_mag_iti, '-o', label='ITI')
    plt.plot(dates, mean_mag_soho, '-o', label='SOHO')
    plt.legend()
    plt.savefig(os.path.join(base_path, 'evaluation/mag_comparison.jpg'), dpi=300)
    plt.close()