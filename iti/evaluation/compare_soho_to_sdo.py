import glob
import os
from datetime import timedelta
from warnings import simplefilter

from aiapy.calibrate import correct_degradation
from astropy import units as u
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from matplotlib.cm import get_cmap
from skimage.io import imsave
from sunpy.map import Map, all_coordinates_from_map
from sunpy.map.sources import AIAMap
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from iti.data.editor import sdo_norms, soho_norms, get_local_correction_table

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from iti.translate import SOHOToSDO

import numpy as np

# init
base_path = "/gss/r.jarolim/iti/soho_sdo_v25"
prediction_path = os.path.join(base_path, 'compare')
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
                  np.abs(date_soho - date_sdo) < timedelta(hours=1)] # file name filter (below filter < 1 min)

basenames_soho = ['%s.fits' % date_soho.isoformat('T') for date_soho, date_sdo in selected_dates]
basenames_sdo = ['%s.fits' % date_sdo.isoformat('T') for date_soho, date_sdo in selected_dates]

iti_maps = translator.translate('/gss/r.jarolim/data/soho_iti2021_prep', basenames=basenames_soho)
soho_maps = ([Map('/gss/r.jarolim/data/soho_iti2021_prep/%s/%s' % (dir, basename))
              for dir in ['171', '195', '284', '304', 'mag']]
             for basename in basenames_soho)
sdo_maps = ([Map('/gss/r.jarolim/data/sdo_comparison/%s/%s' % (dir, basename))
             for dir in ['171', '193', '211', '304', '6173']]
    for basename in basenames_sdo)

cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    'gray'
]

pos = 1000

for soho_cube, iti_cube, sdo_cube in tqdm(zip(soho_maps, iti_maps, sdo_maps), total=len(selected_dates)):
    date = soho_cube[0].date.datetime
    if np.abs(soho_cube[0].date.datetime - sdo_cube[0].date.datetime) > timedelta(minutes=1):
        print('Invalid!', np.abs(soho_cube[0].date.datetime - sdo_cube[0].date.datetime))
        continue
    print(np.abs(soho_cube[0].date.datetime - sdo_cube[0].date.datetime))
    simplefilter('ignore')  # ignore int conversion warning
    dir = os.path.join(os.path.join(prediction_path, '%s') % date.isoformat())
    if os.path.exists(dir):
        continue
    os.makedirs(dir, exist_ok=True)
    for i, (s_map, cmap, norm) in enumerate(zip(soho_cube, cmaps, soho_norms.values())):
        cmap = get_cmap(cmap) if isinstance(cmap, str) else cmap
        s_map = s_map.rotate(recenter=True)
        bl = SkyCoord(-pos * u.arcsec, -pos * u.arcsec, frame=s_map.coordinate_frame)
        tr = SkyCoord(pos * u.arcsec, pos * u.arcsec, frame=s_map.coordinate_frame)
        s_map = s_map.submap(bl, top_right=tr)
        if i == 4:
            hpc_coords = all_coordinates_from_map(s_map)
            r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
            s_map.data[r > 1] = -1000
        imsave(dir + '/soho_%d.jpg' % s_map.wavelength.value, cmap(norm(s_map.data))[..., :-1], check_contrast=False)
    for i, (s_map, cmap, norm) in enumerate(zip(iti_cube, cmaps,
                                 [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304], sdo_norms['mag']])):
        cmap = get_cmap(cmap) if isinstance(cmap, str) else cmap
        s_map = s_map.rotate(recenter=True)
        bl = SkyCoord(-pos * u.arcsec, -pos * u.arcsec, frame=s_map.coordinate_frame)
        tr = SkyCoord(pos * u.arcsec, pos * u.arcsec, frame=s_map.coordinate_frame)
        s_map = s_map.submap(bl, top_right=tr)
        if i == 4:
            hpc_coords = all_coordinates_from_map(s_map)
            r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
            s_map.data[r > 1] = -1000
        imsave(dir + '/iti_%d.jpg' % s_map.wavelength.value, cmap(norm(s_map.data))[..., :-1], check_contrast=False)
    for i, (s_map, cmap, norm) in enumerate(zip(sdo_cube, cmaps,
                                 [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304], sdo_norms['mag']])):
        cmap = get_cmap(cmap) if isinstance(cmap, str) else cmap
        if isinstance(s_map, AIAMap):
            s_map = correct_degradation(s_map, correction_table=get_local_correction_table())
            data = np.nan_to_num(s_map.data)
            data = data / s_map.meta["exptime"]
            s_map = Map(data, s_map.meta)
        s_map = s_map.rotate(recenter=True, )
        bl = SkyCoord(-pos * u.arcsec, -pos * u.arcsec, frame=s_map.coordinate_frame)
        tr = SkyCoord(pos * u.arcsec, pos * u.arcsec, frame=s_map.coordinate_frame)
        s_map = s_map.submap(bl, top_right=tr)
        if i == 4:
            hpc_coords = all_coordinates_from_map(s_map)
            r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
            s_map.data[r > 1] = -1000
        imsave(dir + '/sdo_%d.jpg' % s_map.wavelength.value, cmap(norm(s_map.data))[..., :-1], check_contrast=False)
