import copy
import glob
import os
from datetime import timedelta, datetime
from warnings import simplefilter

import matplotlib.pyplot as plt
import pylab
from astropy import units as u
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from imreg_dft import similarity, transform_img_dict
from matplotlib.colors import Normalize
from skimage.metrics import structural_similarity
from sunpy.map import Map, all_coordinates_from_map
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.visualization.colormaps import cm
from tqdm import tqdm



from itipy.translate import SOHOToSDO

import numpy as np

import matplotlib.gridspec as gridspec
gridspec
# init
base_path = "/gpfs/gpfs0/robert.jarolim/iti/soho_sdo_v4"
soho_data_path = '/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep'
sdo_data_path = '/gpfs/gpfs0/robert.jarolim/data/iti/sdo_comparison_iti2022'

prediction_path = '/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3'
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = SOHOToSDO(model_path=os.path.join(base_path, 'generator_AB.pt'))

# translate
basenames_soho = [[os.path.basename(f) for f in glob.glob('%s/%s/*.fits' % (soho_data_path, wl))] for
                  wl in ['171', '195', '284', '304', 'mag']]
basenames_soho = set(basenames_soho[0]).intersection(*basenames_soho[1:])
basenames_sdo = [[os.path.basename(f) for f in glob.glob('%s/%s/*.fits' % (sdo_data_path,wl))] for wl in
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

iti_maps = translator.translate(soho_data_path, basenames=basenames_soho)
soho_maps = ([Map('%s/%s/%s' % (soho_data_path, dir, basename))
              for dir in ['171', '195', '284', '304', 'mag']]
             for basename in basenames_soho)
sdo_maps = ([Map('%s/%s/%s' % (sdo_data_path, dir, basename))
             for dir in ['171', '193', '211', '304', '6173']]
    for basename in basenames_sdo)

center_coord = SkyCoord(40 * u.arcsec, -450 * u.arcsec, frame='helioprojective', obstime=datetime(2010, 9, 17, 1), observer='earth')

def get_submap(s_map):
    coord = solar_rotate_coordinate(center_coord, time=sdo_map.date)
    arcs = 60 * u.arcsec
    s_map = s_map.submap(
        bottom_left=SkyCoord(coord.Tx - arcs, coord.Ty - arcs, frame=sdo_map.coordinate_frame),
        top_right=SkyCoord(coord.Tx + arcs, coord.Ty + arcs, frame=sdo_map.coordinate_frame))
    hpc_coords = all_coordinates_from_map(s_map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
    s_map.data[r > 1] = np.nan
    return s_map

def clip(s_map):
    return Map(np.clip(s_map.data, -3000, 3000), s_map.meta)

maps = {i: [] for i in range(5)}
for soho_cube, iti_cube, sdo_cube in tqdm(zip(soho_maps, iti_maps, sdo_maps), total=len(selected_dates)):
    date = soho_cube[-1].date.datetime
    if np.abs(soho_cube[-1].date.datetime - sdo_cube[-1].date.datetime) > timedelta(minutes=10):
        print('Invalid!', np.abs(soho_cube[-1].date.datetime - sdo_cube[-1].date.datetime))
        continue
    simplefilter('ignore')  # ignore int conversion warning
    for i in range(5):
        sdo_map = sdo_cube[i].rotate(recenter=True)
        soho_map = soho_cube[i].rotate(recenter=True)
        iti_map = iti_cube[i]
        #
        sdo_map = clip(get_submap(sdo_map))
        soho_map = clip(get_submap(soho_map))
        iti_map = clip(get_submap(iti_map))
        #
        maps[i].append((soho_map, iti_map, sdo_map))

def register(map1, map2, missing_val=np.nan):
    d_1 = map1.resample(map2.data.shape * u.pix, 'spline').data
    d_2 = map2.data
    shape = (min(d_1.shape[0], d_2.shape[0]), min(d_1.shape[1], d_2.shape[1]))
    d_1, d_2 = d_1[:shape[0], :shape[1]], d_2[:shape[0], :shape[1]]

    transformation = similarity(d_2, d_1, numiter=20, constraints={'scale': (1, 0), 'angle': (0, 10)})
    print(transformation['tvec'], transformation['angle'])
    d_1_registered = transform_img_dict(d_1, transformation, bgval=missing_val, order=1)
    return d_1_registered, d_2

def MAE(map1, map2):
    d1, d2 = register(map1, map2)
    return np.nanmean(np.abs(d1 - d2))

params = {'axes.labelsize': 'large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
pylab.rcParams.update(params)

fig = plt.figure(constrained_layout=True, figsize=(8.5, 12))
gs = fig.add_gridspec(5,3)

mag_maps = np.array(maps[4])

cmap = copy.deepcopy(cm.hmimag)
cmap.set_bad('black',1.)
v_min_max = 1500#np.nanmax([np.nanmax(np.abs(s_map.data)) for s_map in mag_maps[:, 2]])

print(v_min_max)
for i, map_cube in enumerate(mag_maps):
    ax = fig.add_subplot(gs[i, -3])
    extent = [map_cube[0].bottom_left_coord.Tx.value, map_cube[0].top_right_coord.Tx.value,
                map_cube[0].bottom_left_coord.Ty.value, map_cube[0].top_right_coord.Ty.value]
    ax.imshow(map_cube[0].data, norm=Normalize(vmin=-v_min_max, vmax=v_min_max), cmap=cmap, extent=extent)
    ax = fig.add_subplot(gs[i, -2])
    extent = [map_cube[1].bottom_left_coord.Tx.value, map_cube[1].top_right_coord.Tx.value,
                map_cube[1].bottom_left_coord.Ty.value, map_cube[1].top_right_coord.Ty.value]
    ax.imshow(map_cube[1].data, norm=Normalize(vmin=-v_min_max, vmax=v_min_max), cmap=cmap, extent=extent)
    ax = fig.add_subplot(gs[i, -1])
    extent = [map_cube[2].bottom_left_coord.Tx.value, map_cube[2].top_right_coord.Tx.value,
                map_cube[2].bottom_left_coord.Ty.value, map_cube[2].top_right_coord.Ty.value]
    ax.imshow(map_cube[2].data, norm=Normalize(vmin=-v_min_max, vmax=v_min_max), cmap=cmap, extent=extent)
    ax.set_ylabel(map_cube[0].date.datetime.strftime('%d-%H:%M'), rotation=-90, fontsize=18, labelpad=20)
    ax.yaxis.set_label_position("right")

fig.tight_layout()
plt.savefig(os.path.join(prediction_path, 'magnetogram_series.jpg'), dpi=300)
plt.close()



fig, axs = plt.subplots(1, 2, figsize=(12, 3))
mdi_ssim = [structural_similarity(*register(map1, map2, 0), data_range=6000) for map1, map2 in zip(mag_maps[:, 0], mag_maps[:, 2])]
axs[0].plot([s_map.date.datetime for s_map in mag_maps[:, 2]],
            mdi_ssim,
        '-o', label='MDI')
iti_ssim = [structural_similarity(*register(map1, map2, 0), data_range=6000) for map1, map2 in zip(mag_maps[:, 1], mag_maps[:, 2])]
axs[0].plot([s_map.date.datetime for s_map in mag_maps[:, 2]],
            iti_ssim,
        '-o', label='ITI')
axs[0].set_ylabel('SSIM')

mdi_mae = [MAE(map1, map2) for map1, map2 in zip(mag_maps[:, 0], mag_maps[:, 2])]
axs[1].plot([s_map.date.datetime for s_map in mag_maps[:, 2]],
            mdi_mae,
        '-o', label='MDI')
iti_mae = [MAE(map1, map2) for map1, map2 in zip(mag_maps[:, 1], mag_maps[:, 2])]
axs[1].plot([s_map.date.datetime for s_map in mag_maps[:, 2]],
            iti_mae,
        '-o', label='ITI')
axs[1].legend()
axs[1].set_ylabel('MAE [Gauss]')

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(prediction_path, 'magnetogram_similarity.png'), dpi=300, transparent=True)
plt.close()

print('MEAN SSIM: ITI %.03f; MDI %.03f' % (np.mean(iti_ssim), np.mean(mdi_ssim)))
print('MEAN MAE: ITI %.03f; MDI %.03f' % (np.mean(iti_mae), np.mean(mdi_mae)))

eit_calibration = {'171': [113.69278, 40.340622], '195': [60.60053, 31.752682], '284': [4.7249465, 3.9555929], '304': [64.73511, 26.619505]}
aia_calibration = {'171': [148.90274, 62.101795], '193': [146.01889, 71.47675], '211': [44.460854, 27.592617], '304': [46.21493, 18.522688]}

for i, (eit_calib, aia_calib) in enumerate(zip(eit_calibration.values(), aia_calibration.values())):
    s_maps = np.array(maps[i])

    soho_map = s_maps[:, 0]
    eit_mean, eit_std = eit_calib
    aia_mean, aia_std = aia_calib
    soho_map = (soho_map - eit_mean) * (aia_std / eit_std) + aia_mean

    iti_ssim = [structural_similarity(*register(map1, map2, 0), data_range=np.max(map2)) for map1, map2 in
                zip(s_maps[:, 1], s_maps[:, 2])]
    soho_ssim = [structural_similarity(*register(map1, map2, 0), data_range=np.max(map2)) for map1, map2 in
                zip(soho_map, s_maps[:, 2])]
    iti_mae = [MAE(map1, map2) for map1, map2 in zip(s_maps[:, 1], s_maps[:, 2])]
    soho_mae = [MAE(map1, map2) for map1, map2 in zip(soho_map, s_maps[:, 2])]
    print('SSIM %d: ITI %.03f; SOHO %.03f;' % (i, np.mean(iti_ssim), np.mean(soho_ssim)))
    print('MAE %d: ITI %.03f; SOHO %.03f;' % (i, np.mean(iti_mae), np.mean(soho_mae)))