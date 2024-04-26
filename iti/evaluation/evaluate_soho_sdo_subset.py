import copy
import gc
import glob
import os
import pickle
from datetime import timedelta, datetime
from warnings import simplefilter

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pylab
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from matplotlib.colors import Normalize
from sunpy.map import Map, all_coordinates_from_map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from iti.translate import SOHOToSDO

gridspec
# init
base_path = "/beegfs/home/robert.jarolim/iti_evaluation/mdi_hmi"
soho_data_path = '/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep'
sdo_data_path = '/gpfs/gpfs0/robert.jarolim/data/iti/sdo_comparison_iti2022'
prediction_path = os.path.join(base_path, 'evaluation')
os.makedirs(prediction_path, exist_ok=True)

result_pickle = os.path.join(prediction_path, 'unsigned_flux.pickle')
# create translator
translator = SOHOToSDO(model_path='/gpfs/gpfs0/robert.jarolim/iti/soho_sdo_v4/generator_AB.pt')

# translate
basenames_soho = [[os.path.basename(f) for f in glob.glob('%s/%s/*.fits' % (soho_data_path, wl))] for
                  wl in ['171', '195', '284', '304', 'mag']]
basenames_soho = set(basenames_soho[0]).intersection(*basenames_soho[1:])
basenames_sdo = [[os.path.basename(f) for f in glob.glob('%s/%s/*.fits' % (sdo_data_path, wl))] for wl in
                 ['171', '193', '211', '304', '6173']]
basenames_sdo = set(basenames_sdo[0]).intersection(*basenames_sdo[1:])

dates_soho = sorted([parse(f.split('.')[0]) for f in basenames_soho])
dates_sdo = sorted([parse(f.split('.')[0]) for f in basenames_sdo])

closest_dates = [(date_soho, min(dates_sdo, key=lambda x: abs(x - date_soho))) for date_soho in dates_soho]
selected_dates = [(date_soho, date_sdo) for date_soho, date_sdo in closest_dates if
                  np.abs(date_soho - date_sdo) < timedelta(hours=1)]  # file name filter (below filter < 1 h)

basenames_soho = ['%s.fits' % date_soho.isoformat('T') for date_soho, date_sdo in selected_dates]
basenames_sdo = ['%s.fits' % date_sdo.isoformat('T') for date_soho, date_sdo in selected_dates]

lcs = []
dates = []

iti_maps = translator.translate(soho_data_path, basenames=basenames_soho)
soho_maps = ([Map('%s/%s/%s' % (soho_data_path, dir, basename)) for dir in ['mag']] for basename in basenames_soho)
sdo_maps = ([Map('%s/%s/%s' % (sdo_data_path, dir, basename)) for dir in ['6173']] for basename in basenames_sdo)

def get_submap(s_map):
    arcs = s_map.rsun_obs * 1.0
    s_map = s_map.submap(
        bottom_left=SkyCoord(-arcs, -arcs, frame=s_map.coordinate_frame),
        top_right=SkyCoord(arcs, arcs, frame=s_map.coordinate_frame))
    hpc_coords = all_coordinates_from_map(s_map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
    s_map.data[r > 1] = np.nan
    return s_map


def clip(s_map):
    data = s_map.data
    data[data < -3000] = -3000
    data[data > 3000] = 3000
    return Map(data, s_map.meta)


cmap = copy.deepcopy(cm.hmimag)
# cmap.set_bad('black',1.)

for i, (soho_cube, iti_cube, sdo_cube) in tqdm(enumerate(zip(soho_maps, iti_maps, sdo_maps)),
                                               total=len(selected_dates)):
    date = soho_cube[-1].date.to_datetime()
    if np.abs(soho_cube[0].date.datetime - sdo_cube[0].date.datetime) > timedelta(minutes=10):
        print('Invalid!', np.abs(soho_cube[0].date.datetime - sdo_cube[0].date.datetime))
        continue
    if date in dates:
        print('Already evaluated!')
        continue
    simplefilter('ignore')  # ignore int conversion warning
    hmi_map = sdo_cube[-1].rotate(recenter=True, missing=0, order=4)
    mdi_map = soho_cube[-1]  # .rotate(recenter=True, missing=0, order=4)
    iti_map = iti_cube[-1]
    #
    hmi_map = clip(get_submap(hmi_map))
    # mdi_map = clip(get_submap(mdi_map))
    iti_map = clip(get_submap(iti_map))
    #
    lcs.append((np.nanmean(np.abs(mdi_map.data)), np.nanmean(np.abs(iti_map.data)), np.nanmean(np.abs(hmi_map.data))))
    dates.append((mdi_map.date.to_datetime(), iti_map.date.to_datetime(), hmi_map.date.to_datetime()))
    if (i + 1) % 5 == 0:
        norm = Normalize(vmin=-1500.0, vmax=1500.0)
        plt.imsave(os.path.join(prediction_path, '%s_MDI.jpg' % date.isoformat('T')), norm(mdi_map.data), cmap=cmap,
                   vmin=0, vmax=1, origin='lower')
        plt.imsave(os.path.join(prediction_path, '%s_ITI.jpg' % date.isoformat('T')), norm(iti_map.data), cmap=cmap,
                   vmin=0, vmax=1, origin='lower')
        plt.imsave(os.path.join(prediction_path, '%s_HMI.jpg' % date.isoformat('T')), norm(hmi_map.data), cmap=cmap,
                   vmin=0, vmax=1, origin='lower')
        gc.collect()

        print(mdi_map.date)

        with open(result_pickle, 'wb') as f:
            pickle.dump({'dates': dates, 'lcs': lcs}, f)

with open(result_pickle, 'wb') as f:
    pickle.dump({'dates': dates, 'lcs': lcs}, f)

lcs_arr = np.array(lcs)
dates_arr = np.array(dates)

mae_iti = np.abs(lcs_arr[:, 1] - lcs_arr[:, 2]).mean()
mae_mdi = np.abs(lcs_arr[:, 0] - lcs_arr[:, 2]).mean()
cc_iti = np.corrcoef(lcs_arr[:, 1], lcs_arr[:, 2])[0, 1]
cc_mdi = np.corrcoef(lcs_arr[:, 0], lcs_arr[:, 2])[0, 1]
print('MEAN MAE: ITI %.03f; MDI %.03f' % (mae_iti, mae_mdi))
print('CC: ITI %.03f; MDI %.03f' % (cc_iti, cc_mdi))

# create reference colormap
mpb = plt.imshow(hmi_map.data, cmap=cmap, vmin=-1500.0, vmax=1500.0, origin='lower')
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
cbar = plt.colorbar(mpb, ax=ax, label='[Gauss]')
# cbar.formatter.set_powerlimits((0, 0))
# cbar.set_ticks([-1500, 500, 0, 500, 1500])
ax.remove()
fig.savefig(os.path.join(prediction_path, 'mag_colorbar.png'), dpi=300, transparent=True)
plt.close(fig)

params = {'axes.labelsize': 'x-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'large'}
pylab.rcParams.update(params)
plt.rc('legend', fontsize='x-large')

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(dates_arr[:, 0], lcs_arr[:, 0], '-o', label='MDI (MAE: %0.2f; CC: %0.2f)' % (mae_mdi, cc_mdi))
ax.plot(dates_arr[:, 1], lcs_arr[:, 1], '-o', label='ITI (MAE: %0.2f; CC: %0.2f)' % (mae_iti, cc_iti))
ax.plot(dates_arr[:, 2], lcs_arr[:, 2], '-o', label='HMI')
ax.legend()
ax.set_ylabel('Flux Density [Gauss]')

ax.axvspan(datetime(2010, 9, 13), datetime(2010, 9, 20), facecolor='b', alpha=0.3)

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig(os.path.join(prediction_path, 'full_soho_unsigned.jpg'), dpi=300)
plt.close()
