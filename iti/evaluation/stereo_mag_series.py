import glob
import os
from datetime import datetime

from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from sunpy.coordinates import Helioprojective
from sunpy.physics.differential_rotation import solar_rotate_coordinate

from sunpy.map import Map, all_coordinates_from_map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from iti.translate import STEREOToSDOMagnetogram

from astropy import units as u

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from matplotlib import pyplot as plt

import numpy as np

base_path = "/gss/r.jarolim/iti/stereo_mag_v11"
prediction_path = os.path.join(base_path, 'evaluation')
os.makedirs(prediction_path, exist_ok=True)

soho_basenames = np.array([os.path.basename(f) for f in sorted(glob.glob('/gss/r.jarolim/data/soho_iti2021_prep/304/*.fits'))])
soho_dates = np.array([parse(bn.split('.')[0]) for bn in soho_basenames])
stereo_basenames = np.array(
    [os.path.basename(f) for f in sorted(glob.glob('/gss/r.jarolim/data/stereo_iti2021_prep/304/*.fits'))])
stereo_dates = np.array([parse(bn.split('.')[0]) for bn in stereo_basenames])

min_diff = np.array([np.min(np.abs(stereo_date - soho_dates)) for stereo_date in stereo_dates])
# filter time diff
cond = (stereo_dates <= datetime(2007, 1, 23)) & (stereo_dates >= datetime(2007, 1, 14))
stereo_dates = stereo_dates[cond]
stereo_basenames = stereo_basenames[cond]
# select corresponding stereo files
soho_basenames = soho_basenames[[np.argmin(np.abs(stereo_date - soho_dates)) for stereo_date in stereo_dates]]

stereo_dataset = [[Map("/gss/r.jarolim/data/stereo_iti2021_prep/%s/%s" % (wl, bn)) for wl in ['171', '195', '284', '304']] for bn in stereo_basenames]

soho_dataset = [[Map("/gss/r.jarolim/data/soho_iti2021_prep/%s/%s" % (wl, bn)) for wl in ['171', '195', '284', '304', 'mag']] for bn in soho_basenames]
translator = STEREOToSDOMagnetogram(model_path=os.path.join(base_path, 'generator_AB.pt'))

result = translator.translate("/gss/r.jarolim/data/stereo_iti2021_prep", basenames=stereo_basenames)

sdo_cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    plt.get_cmap('gray')
]

soho_mag_maps = []
iti_mag_maps = []
stereo_maps = []

for date, (iti_map_cube, stereo_img, iti_img), soho_map_cube, stereo_map_cube in tqdm(zip(soho_dates, result, soho_dataset, stereo_dataset), total=len(stereo_basenames)):
    soho_mag = soho_map_cube[-1].rotate(recenter=True)
    iti_mag = iti_map_cube[-1].rotate(recenter=True)
    stereo_map = stereo_map_cube[-1].rotate(recenter=True)
    #
    soho_mag_maps.append(soho_mag)
    iti_mag_maps.append(iti_mag)
    stereo_maps.append(stereo_map)


iti_flux = []
soho_flux = []

iti_dates = np.array([m.date.datetime for m in iti_mag_maps])
soho_dates = np.array([m.date.datetime for m in soho_mag_maps])

fig = plt.figure(constrained_layout=True, figsize=(len(iti_dates) * 3, 12))
gs = fig.add_gridspec(5, len(iti_dates) + 2)

ref_coord = SkyCoord(-880 * u.arcsec, 100 * u.arcsec, frame=Helioprojective, obstime=datetime(2007, 1, 14), observer='earth')

for i, s_map in enumerate(stereo_maps):
    box_size = s_map.rsun_obs * 0.13
    center_coord = solar_rotate_coordinate(ref_coord, observer=s_map.observer_coordinate)
    bl = SkyCoord(center_coord.Tx - box_size, center_coord.Ty - box_size, frame=s_map.coordinate_frame)
    tr = SkyCoord(center_coord.Tx + box_size, center_coord.Ty + box_size, frame=s_map.coordinate_frame)
    s_map = s_map.submap(bottom_left=bl, top_right=tr)
    #
    ax = fig.add_subplot(gs[2, 2+i])
    s_map.plot(axes=ax, cmap=cm.sdoaia304, vmin=0, vmax=18100)
    ax.set_title(s_map.date.datetime.isoformat(' ', timespec='hours')[:-3], fontsize=20)
    ax.set_xlabel('')
    ax.set_ylabel('') if i != 0 else ax.set_ylabel('STEREO 304', fontsize=20)

for i, s_map in enumerate(iti_mag_maps):
    box_size = s_map.rsun_obs * 0.13
    center_coord = solar_rotate_coordinate(ref_coord, observer=s_map.observer_coordinate)
    bl = SkyCoord(center_coord.Tx - box_size, center_coord.Ty - box_size, frame=s_map.coordinate_frame)
    tr = SkyCoord(center_coord.Tx + box_size, center_coord.Ty + box_size, frame=s_map.coordinate_frame)
    s_map = s_map.submap(bottom_left=bl, top_right=tr)
    hpc_coords = all_coordinates_from_map(s_map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
    s_map.data[r > 1] = np.nan
    #
    ax = fig.add_subplot(gs[3, 2+i])
    #ax.set_axis_off()
    s_map.plot(axes=ax, cmap=cm.hmimag, vmin=-1500, vmax=1500, title='')
    ax.set_xlabel('')
    ax.set_ylabel('') if i != 0 else ax.set_ylabel('ITI', fontsize=20)
    iti_flux.append(np.nanmean(np.abs(s_map.data)))
    #ax.imshow(np.abs(s_map.data), cmap=cm.hmimag, vmin=-1500, vmax=1500)

for i, s_map in enumerate(soho_mag_maps):
    box_size = s_map.rsun_obs * 0.13
    center_coord = solar_rotate_coordinate(ref_coord, observer=s_map.observer_coordinate)
    bl = SkyCoord(center_coord.Tx - box_size, center_coord.Ty - box_size, frame=s_map.coordinate_frame)
    tr = SkyCoord(center_coord.Tx + box_size , center_coord.Ty + box_size, frame=s_map.coordinate_frame)
    s_map = s_map.submap(bottom_left=bl, top_right=tr)
    hpc_coords = all_coordinates_from_map(s_map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
    s_map.data[r > 1] = np.nan
    #
    ax = fig.add_subplot(gs[4, 2+i])
    #ax.set_axis_off()
    s_map = Map(np.abs(s_map.data), s_map.meta)
    s_map.plot(axes=ax, cmap=cm.hmimag, vmin=-1500, vmax=1500, title='')
    ax.set_xlabel('')
    ax.set_ylabel('') if i != 0 else ax.set_ylabel('SOHO/MDI', fontsize=20)
    #ax.imshow(np.abs(s_map.data), cmap=cm.hmimag, vmin=-1500, vmax=1500)
    soho_flux.append(np.nanmean(np.abs(s_map.data)))

ax = fig.add_subplot(gs[:2, 2:])
ax.plot(iti_dates, iti_flux, '-o', label='ITI')
ax.plot(soho_dates, soho_flux, '-o', label='SOHO')
ax.fill_between([m.date.datetime for m in soho_mag_maps], np.array(soho_flux)-15, np.array(soho_flux)+15, alpha=0.2, color='C1')
ax.set_ylabel('Mean Magnetic Flux [Gauss]', fontsize=20)
ax.legend(fontsize='xx-large')

ax = fig.add_subplot(gs[:2, :2])
s_map =stereo_maps[4]
bl = SkyCoord(- 1.1 * s_map.rsun_obs, - 1.1 * s_map.rsun_obs, frame=s_map.coordinate_frame)
tr = SkyCoord(1.1 * s_map.rsun_obs, 1.1 * s_map.rsun_obs, frame=s_map.coordinate_frame)
s_map = s_map.submap(bl, tr)
box_size = s_map.rsun_obs * 0.13
center_coord = solar_rotate_coordinate(ref_coord, observer=s_map.observer_coordinate)
bl = SkyCoord(center_coord.Tx - box_size, center_coord.Ty - box_size, frame=s_map.coordinate_frame)
tr = SkyCoord(center_coord.Tx + box_size, center_coord.Ty + box_size, frame=s_map.coordinate_frame)
s_map.plot(axes=ax, cmap=cm.sdoaia304, vmin=0, vmax=18100)
ax.set_title(s_map.date.datetime.isoformat(' ', timespec='hours')[:-3], fontsize=25)
s_map.draw_rectangle(axes=ax, bottom_left=bl, top_right=tr, color='red', linewidth=5)
ax.set_axis_off()

ax = fig.add_subplot(gs[2:4, :2])
s_map = iti_mag_maps[4]
bl = SkyCoord(- 1.1 * s_map.rsun_obs, - 1.1 * s_map.rsun_obs, frame=s_map.coordinate_frame)
tr = SkyCoord(1.1 * s_map.rsun_obs, 1.1 * s_map.rsun_obs, frame=s_map.coordinate_frame)
s_map = s_map.submap(bl, tr)
hpc_coords = all_coordinates_from_map(s_map)
r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
s_map.data[r > 1] = np.nan
box_size = s_map.rsun_obs * 0.13
center_coord = solar_rotate_coordinate(ref_coord, observer=s_map.observer_coordinate)
bl = SkyCoord(center_coord.Tx - box_size, center_coord.Ty - box_size, frame=s_map.coordinate_frame)
tr = SkyCoord(center_coord.Tx + box_size, center_coord.Ty + box_size, frame=s_map.coordinate_frame)
s_map.plot(axes=ax, cmap=cm.hmimag, vmin=-1500, vmax=1500, title='')
ax.set_title('ITI full-disk', fontsize=25)
s_map.draw_rectangle(axes=ax, bottom_left=bl, top_right=tr, color='red', linewidth=5)
ax.set_axis_off()


#fig.tight_layout()
fig.savefig(os.path.join(prediction_path, 'mag_comparison.jpg'), dpi=300)
plt.close()
