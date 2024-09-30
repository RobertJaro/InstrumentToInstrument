import glob
import os
from warnings import simplefilter

from astropy import units as u
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from matplotlib.cm import get_cmap
from skimage.io import imsave
from sunpy.map import Map, all_coordinates_from_map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from itipy.data.editor import sdo_norms, soho_norms



from itipy.translate import SOHOToSDO

import numpy as np

# init
base_path = "/gpfs/gpfs0/robert.jarolim/iti/soho_sdo_v4"
prediction_path = os.path.join(base_path, 'plot')
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = SOHOToSDO(model_path=os.path.join(base_path, 'generator_AB.pt'))

# translate
data_path = '/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep'
basenames_soho = [[os.path.basename(f) for f in glob.glob(os.path.join(data_path, wl, '*.fits'))]
                  for wl in ['171', '195', '284', '304', 'mag']]

basenames_soho = np.array(sorted(list(set(basenames_soho[0]).intersection(*basenames_soho[1:]))))
dates_soho = np.array([parse(f.split('.')[0]) for f in basenames_soho])

cond = [d.month in [11, 12] for d in dates_soho]
dates_soho = dates_soho[cond][::100]
basenames_soho = basenames_soho[cond][::100]

iti_maps = translator.translate(data_path, basenames=basenames_soho)
def loadMap(f):
    s_map = Map(f)
    s_map.meta['DATE-OBS'] = s_map.meta['DATE_OBS']  # fix date
    return s_map

soho_maps = ([loadMap(glob.glob(os.path.join(data_path, dir, basename)))
              for dir in ['171', '195', '284', '304', 'mag']]
             for basename in basenames_soho)

cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    'gray'
]

pos = 1000

for soho_cube, iti_cube in tqdm(zip(soho_maps, iti_maps), total=len(dates_soho)):
    date = soho_cube[0].date.datetime
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
            s_map.data[r > 1] = -3000
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