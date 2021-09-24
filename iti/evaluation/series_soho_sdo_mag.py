import glob
import os
from datetime import timedelta, datetime
from warnings import simplefilter

import matplotlib.pyplot as plt
import pylab
from aiapy.calibrate import correct_degradation
from astropy import units as u
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from skimage.io import imsave
from skimage.metrics import structural_similarity
from sunpy.map import Map, all_coordinates_from_map
from sunpy.map.sources import AIAMap, MDIMap
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from iti.data.editor import sdo_norms, soho_norms, get_local_correction_table

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from iti.prediction.translate import SOHOToSDO

import numpy as np

import matplotlib.gridspec as gridspec
gridspec
# init
base_path = "/gss/r.jarolim/iti/soho_sdo_v25"
prediction_path = os.path.join(base_path, 'evaluation')
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = SOHOToSDO(model_path=os.path.join(base_path, 'generator_AB.pt'))

# translate
basenames_soho = [[os.path.basename(f) for f in glob.glob('/gss/r.jarolim/data/soho_iti2021_prep/%s/*.fits' % wl)] for
                  wl in ['171', '195', '284', '304', 'mag']]
basenames_soho = set(basenames_soho[0]).intersection(*basenames_soho[1:])
basenames_sdo = [[os.path.basename(f) for f in glob.glob('/gss/r.jarolim/data/sdo_comparison/%s/*.fits' % wl)] for wl in
                 ['171', '193', '211', '304', '6173']]
basenames_sdo = set(basenames_sdo[0]).intersection(*basenames_sdo[1:])

dates_soho = sorted([parse(f.split('.')[0]) for f in basenames_soho])
dates_sdo = sorted([parse(f.split('.')[0]) for f in basenames_sdo])

closest_dates = [(date_soho, min(dates_sdo, key=lambda x: abs(x - date_soho))) for date_soho in dates_soho]
selected_dates = [(date_soho, date_sdo) for date_soho, date_sdo in closest_dates if
                  np.abs(date_soho - date_sdo) < timedelta(hours=1)
                  and date_soho > datetime(2010, 9, 13) and date_soho < datetime(2010, 9, 20)] # file name filter (below filter < 1 min)

basenames_soho = ['%s.fits' % date_soho.isoformat('T') for date_soho, date_sdo in selected_dates]
basenames_sdo = ['%s.fits' % date_sdo.isoformat('T') for date_soho, date_sdo in selected_dates]

iti_maps = translator.translate('/gss/r.jarolim/data/soho_iti2021_prep', basenames=basenames_soho)
soho_maps = ([Map('/gss/r.jarolim/data/soho_iti2021_prep/%s/%s' % (dir, basename))
              for dir in ['171', '195', '284', '304', 'mag']]
             for basename in basenames_soho)
sdo_maps = ([Map('/gss/r.jarolim/data/sdo_comparison/%s/%s' % (dir, basename))
             for dir in ['171', '193', '211', '304', '6173']]
    for basename in basenames_sdo)

center_coord = SkyCoord(50 * u.arcsec, -450 * u.arcsec, frame='helioprojective', obstime=datetime(2010, 9, 17, 1), observer='earth')

def get_submap(s_map):
    coord = solar_rotate_coordinate(center_coord, time=hmi_map.date)
    s_map = s_map.submap(
        bottom_left=SkyCoord(coord.Tx - 75 * u.arcsec, coord.Ty - 75 * u.arcsec, frame=hmi_map.coordinate_frame),
        top_right=SkyCoord(coord.Tx + 75 * u.arcsec, coord.Ty + 75 * u.arcsec, frame=hmi_map.coordinate_frame))
    hpc_coords = all_coordinates_from_map(s_map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
    s_map.data[r > 1] = np.nan
    return s_map

def clip(s_map):
    return Map(np.clip(s_map.data, -1000, 1000), s_map.meta)

maps = []
for soho_cube, iti_cube, sdo_cube in tqdm(zip(soho_maps, iti_maps, sdo_maps), total=len(selected_dates)):
    date = soho_cube[0].date.datetime
    if np.abs(soho_cube[0].date.datetime - sdo_cube[0].date.datetime) > timedelta(minutes=1):
        print('Invalid!', np.abs(soho_cube[0].date.datetime - sdo_cube[0].date.datetime))
        continue
    simplefilter('ignore')  # ignore int conversion warning
    hmi_map = sdo_cube[-1].rotate(recenter=True)
    mdi_map = soho_cube[-1].rotate(recenter=True)
    iti_map = iti_cube[-1]
    #
    hmi_map = clip(get_submap(hmi_map))
    mdi_map = clip(get_submap(mdi_map))
    iti_map = clip(get_submap(iti_map))
    #
    maps.append((mdi_map, iti_map, hmi_map))

maps = np.array(maps)

def adjust_dim(map1, map2):
    d_1 = map1.resample(map2.data.shape * u.pix, 'spline').data
    d_2 = map2.data
    shape = (min(d_1.shape[0], d_2.shape[0]), min(d_1.shape[1], d_2.shape[1]))
    return d_1[:shape[0], :shape[1]], d_2[:shape[0], :shape[1]]

def MAE(map1, map2):
    d1, d2 = adjust_dim(map1, map2)
    return np.mean(np.abs(d1 - d2))

params = {'axes.labelsize': 'large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
pylab.rcParams.update(params)

fig = plt.figure(constrained_layout=True, figsize=(12, 7))
gs = fig.add_gridspec(5, 9)

ax = fig.add_subplot(gs[:2, :])
ax.plot([s_map.date.datetime for s_map in maps[:, 0]], [np.nanmean(np.abs(s_map.data)) for s_map in maps[:, 0]], '-o', label='MDI')
ax.plot([s_map.date.datetime for s_map in maps[:, 1]], [np.nanmean(np.abs(s_map.data)) for s_map in maps[:, 1]], '-o', label='ITI')
ax.plot([s_map.date.datetime for s_map in maps[:, 2]], [np.nanmean(np.abs(s_map.data)) for s_map in maps[:, 2]], '-o', label='HMI')
ax.legend()
ax.set_ylabel('Magnetic Flux [Gauss]')

for i, map_cube in enumerate(maps):
    ax = fig.add_subplot(gs[-3, i])
    map_cube[0].plot(axes=ax, title=False, annotate=False, norm=Normalize(vmin=-1000, vmax=1000))
    ax.set_axis_off()
    ax = fig.add_subplot(gs[-2, i])
    map_cube[1].plot(axes=ax, title=False, annotate=False, norm=Normalize(vmin=-1000, vmax=1000))
    ax.set_axis_off()
    ax = fig.add_subplot(gs[-1, i])
    map_cube[2].plot(axes=ax, title=False, annotate=False, norm=Normalize(vmin=-1000, vmax=1000))
    ax.set_axis_off()

plt.savefig(os.path.join(prediction_path, 'magnetogram_series.jpg'), dpi=300)
plt.close()



fig, axs = plt.subplots(1, 2, figsize=(12, 3))
mdi_ssim = [structural_similarity(*adjust_dim(map1, map2), data_range=2000) for map1, map2 in zip(maps[:, 0], maps[:, 2])]
axs[0].plot([s_map.date.datetime for s_map in maps[:, 2]],
            mdi_ssim,
        '-o', label='MDI')
iti_ssim = [structural_similarity(*adjust_dim(map1, map2), data_range=2000) for map1, map2 in zip(maps[:, 1], maps[:, 2])]
axs[0].plot([s_map.date.datetime for s_map in maps[:, 2]],
            iti_ssim,
        '-o', label='ITI')
axs[0].set_ylabel('SSIM')

mdi_mae = [MAE(map1, map2) for map1, map2 in zip(maps[:, 0], maps[:, 2])]
axs[1].plot([s_map.date.datetime for s_map in maps[:, 2]],
            mdi_mae,
        '-o', label='MDI')
iti_mae = [MAE(map1, map2) for map1, map2 in zip(maps[:, 1], maps[:, 2])]
axs[1].plot([s_map.date.datetime for s_map in maps[:, 2]],
            iti_mae,
        '-o', label='ITI')
axs[1].legend()
axs[1].set_ylabel('MAE [Gauss]')

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(prediction_path, 'magnetogram_similarity.jpg'), dpi=300)
plt.close()

print('MEAN SSIM: ITI %.03f; MDI %.03f' % (np.mean(iti_ssim), np.mean(mdi_ssim)))
print('MEAN MSE: ITI %.03f; MDI %.03f' % (np.mean(iti_mae), np.mean(mdi_mae)))