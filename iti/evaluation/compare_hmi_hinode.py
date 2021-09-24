import glob
import os
from datetime import timedelta, datetime

from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from matplotlib.cm import get_cmap
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from sunpy.map import Map

from iti.data.align import alignMaps
from iti.prediction.translate import HMIToHinode
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from astropy import units as u
import matplotlib.gridspec as gridspec

gridspec

# Functions
base_path = '/gss/r.jarolim/iti/hmi_hinode_v13'
evaluation_path = os.path.join(base_path, "compare")
data_path = '/gss/r.jarolim/data/hmi_hinode_comparison'
os.makedirs(evaluation_path, exist_ok=True)

df = pd.read_csv('/gss/r.jarolim/data/hinode/file_list.csv', index_col=False, parse_dates=['date'])
test_df = df[df.date.dt.month.isin([11, 12])]
test_df = test_df[test_df.classification == 'feature']

hinode_paths = test_df.file
hinode_dates = [Map(f).date.datetime for f  in hinode_paths]

hmi_paths = np.array(sorted(glob.glob(os.path.join(data_path, '6173/*.fits'))))
hmi_dates = np.array([parse(os.path.basename(f).split('.')[0]) for f in hmi_paths])

hmi_paths = hmi_paths[[np.argmin(np.abs(hmi_dates - d)) for d in hinode_dates]]
hmi_dates = np.array([parse(os.path.basename(f).split('.')[0]) for f in hmi_paths])

cond = (np.abs(hmi_dates - hinode_dates) < timedelta(seconds=10)) #& (np.abs(hmi_dates - datetime(2011,11,6,10)) < timedelta(days=1))
hmi_paths = hmi_paths[cond]
hinode_paths = hinode_paths[cond]

translator = HMIToHinode(model_path=os.path.join(base_path, 'generator_AB.pt'), patch_factor=3)

# init maps generator
hinode_maps = (Map(path) for path in hinode_paths)
hmi_maps = (Map(path) for path in hmi_paths)
iti_maps = translator.translate(hmi_paths)


def normalize(data):
    data = (data - np.nanmean(data)) / np.nanstd(data)
    # norm = ImageNormalize(data)
    return data  # norm(data, clip=True).data


result_maps = []
for hmi_map, iti_map, hinode_map in tqdm(zip(hmi_maps, iti_maps, hinode_maps), total=len(hinode_paths)):
    # rescale, rotate, normalize and crop hinode map
    scale = (0.15 * u.arcsec / u.pix)
    hinode_map = hinode_map.rotate(scale=hinode_map.scale[0] / scale, missing=np.nan)
    coord = hinode_map.center
    hinode_map = hinode_map.submap(
        bottom_left=SkyCoord(coord.Tx - 50 * u.arcsec, coord.Ty - 50 * u.arcsec, frame=hinode_map.coordinate_frame),
        top_right=SkyCoord(coord.Tx + 50 * u.arcsec, coord.Ty + 50 * u.arcsec, frame=hinode_map.coordinate_frame))
    hinode_data = hinode_map.data / hinode_map.exposure_time.to(u.s).value * (1 * u.s)
    hinode_map = Map(hinode_data, hinode_map.meta)
    # rotate hmi map
    hmi_map = hmi_map.rotate(recenter=True)
    # rotate iti map
    iti_map = iti_map.rotate(recenter=True)
    #
    # align hmi original map with hinode
    aligned_hinode_map = alignMaps(hinode_map, hmi_map)
    bl = aligned_hinode_map.bottom_left_coord
    tr = aligned_hinode_map.top_right_coord
    width, height = aligned_hinode_map.dimensions
    #
    hinode_data = aligned_hinode_map.data
    hmi_sub_map = hmi_map.submap(bottom_left=bl, top_right=tr).resample(u.Quantity((width, height), u.pix))
    iti_sub_map = iti_map.submap(bottom_left=bl, top_right=tr).resample(u.Quantity((width, height), u.pix))
    hmi_sub_map.data[np.isnan(hinode_data)] = np.nan
    iti_sub_map.data[np.isnan(hinode_data)] = np.nan
    result_maps.append([hmi_sub_map.data, iti_sub_map.data, aligned_hinode_map.data])
    #
    cmap = get_cmap('gray')
    cmap.set_bad(color='black')
    #
    # vmin = np.nanmin([normalize(aligned_hinode_map.data), normalize(hmi_sub_map.data), normalize(iti_sub_map.data)])
    # vmax = np.nanmax([normalize(aligned_hinode_map.data), normalize(hmi_sub_map.data), normalize(iti_sub_map.data)])
    bins = np.linspace(-4, 4, 250)
    #
    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    gs = fig.add_gridspec(5, 9)
    #
    ax = fig.add_subplot(gs[:4, :3])
    hmi_sub_map.plot(axes=ax, cmap='gray')
    ax.set_title('HMI', fontsize=20)
    ax.set_xlabel('Helioprojective Longitude [arcsec]')
    ax.set_ylabel('Helioprojective Latitude [arcsec]')
    ax = fig.add_subplot(gs[4, :])
    ax.hist(normalize(np.ravel(hmi_sub_map.data)), bins, alpha=0.5, label='HMI')
    ax.hist(normalize(np.ravel(iti_sub_map.data)), bins, alpha=0.5, label='ITI')
    ax.hist(normalize(np.ravel(aligned_hinode_map.data)), bins, alpha=0.5, label='Hinode')
    ax.legend()
    #
    ax = fig.add_subplot(gs[:4, 3:6])
    iti_sub_map.plot(axes=ax, cmap='gray')
    ax.set_title('ITI', fontsize=20)
    ax.set_xlabel('Helioprojective Longitude [arcsec]')
    ax.set_ylabel(None)
    #
    ax = fig.add_subplot(gs[:4, 6:9])
    aligned_hinode_map.plot(axes=ax, cmap='gray')
    ax.set_title('Hinode', fontsize=20)
    ax.set_xlabel('Helioprojective Longitude [arcsec]')
    ax.set_ylabel(None)
    #
    fig.tight_layout()
    plt.savefig(os.path.join(evaluation_path, '%s.jpg' % hinode_map.date.datetime.isoformat('T')), dpi=300)
    plt.close(fig)


def intersection(h1, h2):
    return sum([min(b1, b2) for b1, b2 in zip(h1, h2)]) / sum(h1)


maps = np.array([r for r in result_maps if r[0].shape == (668, 668)])

print('NORMALIZED')
bins = np.linspace(-4, 4, 250)
histograms_hmi = [np.histogram(normalize(d), bins)[0] for d in maps[:, 0]]
histograms_iti = [np.histogram(normalize(d), bins)[0] for d in maps[:, 1]]
histograms_hinode = [np.histogram(normalize(d), bins)[0] for d in maps[:, 2]]
print('Intersection ITI-Hinode:', np.mean([intersection(h1, h2) for h1, h2 in zip(histograms_iti, histograms_hinode)]))
# Intersection ITI-Hinode: 0.908063333982485
print('Intersection HMI-Hinode:', np.mean([intersection(h1, h2) for h1, h2 in zip(histograms_hmi, histograms_hinode)]))
# Intersection HMI-Hinode: 0.7392921949233671


print('NOT-NORMALIZED')
bins = np.linspace(900, 60000, 250)
histograms_hmi = [np.histogram(d, bins)[0] for d in maps[:, 0]]
histograms_iti = [np.histogram(d, bins)[0] for d in maps[:, 1]]
histograms_hinode = [np.histogram(d, bins)[0] for d in maps[:, 2]]
print('Intersection ITI-Hinode:', np.mean([intersection(h1, h2) for h1, h2 in zip(histograms_iti, histograms_hinode)]))
# Intersection ITI-Hinode: 0.6414817978649553
print('Intersection HMI-Hinode:', np.mean([intersection(h1, h2) for h1, h2 in zip(histograms_hmi, histograms_hinode)]))
# Intersection HMI-Hinode: 0.06992691179524502

# MSE
# square_error = (hmi_data - hinode_data) ** 2
# axs[1, 0].imshow(square_error, )
# axs[1, 0].set_title('MSE %.03f' % np.nanmean(square_error))
# square_error = (iti_data - hinode_data) ** 2
# axs[1, 1].imshow(square_error, )
# axs[1, 1].set_title('MSE %.03f' % np.nanmean(square_error))
# SSIM
# ssim, ssim_img = structural_similarity(np.nan_to_num(hmi_data, nan=0),
#                                        np.nan_to_num(hinode_data, nan=0), full=True, data_range=1,
#                                        gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
# axs[2, 0].imshow(ssim_img, )
# axs[2, 0].set_title('SSIM %.03f' % ssim)
# ssim, ssim_img = structural_similarity(np.nan_to_num(iti_data, nan=0),
#                                        np.nan_to_num(hinode_data, nan=0), full=True, data_range=1)
# axs[2, 1].imshow(ssim_img, )
# axs[2, 1].set_title('SSIM %.03f' % ssim)
# print('CORRELATION', correlation_coefficient(hmi_data, hinode_data),
#       correlation_coefficient(iti_data, hinode_data))
