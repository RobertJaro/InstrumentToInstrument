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
from sunpy.map import Map
from sunpy.map.sources import AIAMap
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from iti.data.editor import sdo_norms, soho_norms, get_local_correction_table

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from iti.prediction.translate import SOHOToSDO

from matplotlib import pyplot as plt

import numpy as np

# init
base_path = "/gss/r.jarolim/iti/soho_sdo_v24"
prediction_path = os.path.join(base_path, 'evaluation')
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = SOHOToSDO(model_path=os.path.join(base_path, 'generator_AB.pt'))

# translate
basenames_soho = [[os.path.basename(f) for f in glob.glob('/gss/r.jarolim/data/soho_iti2021_prep/%s/*.fits' % wl)] for wl in ['171', '195', '284', '304', 'mag']]
basenames_soho = set(basenames_soho[0]).intersection(*basenames_soho[1:])
basenames_sdo = [[os.path.basename(f) for f in glob.glob('/gss/r.jarolim/data/sdo_comparison/%s/*.fits' % wl)] for wl in ['171', '193', '211', '304', '6173']]
basenames_sdo = set(basenames_sdo[0]).intersection(*basenames_sdo[1:])

dates_soho = sorted([parse(f.split('.')[0]) for f in basenames_soho])
dates_sdo = sorted([parse(f.split('.')[0]) for f in basenames_sdo])

closest_dates = [(date_soho, min(dates_sdo, key=lambda x: abs(x - date_soho))) for date_soho in dates_soho]
selected_dates = [(date_soho, date_sdo) for date_soho, date_sdo in closest_dates if np.abs(date_soho - date_sdo) < timedelta(hours=1)]

basenames_soho = ['%s.fits' % date_soho.isoformat('T') for date_soho, date_sdo in selected_dates]
basenames_sdo = ['%s.fits' % date_sdo.isoformat('T') for date_soho, date_sdo in selected_dates]

iti_maps = translator.translate('/gss/r.jarolim/data/soho_iti2021_prep', basenames=basenames_soho)
soho_maps = [[Map('/gss/r.jarolim/data/soho_iti2021_prep/%s/%s' % (dir, basename)) for dir in ['171', '195', '284', '304', 'mag']]
             for basename in basenames_soho]
sdo_maps = [[Map('/gss/r.jarolim/data/sdo_comparison/%s/%s' % (dir, basename)) for dir in ['171', '193', '211', '304', '6173']]
             for basename in basenames_sdo]
dates = [m[0].date.datetime for m in soho_maps]

cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    'gray'
]

pos = 1000

for soho_cube, iti_cube, sdo_cube, date in tqdm(zip(soho_maps, iti_maps, sdo_maps, dates)):
    simplefilter('ignore') # ignore int conversion warning
    dir = os.path.join(os.path.join(prediction_path, '%s') % date.isoformat())
    os.makedirs(dir, exist_ok=True)
    for s_map, cmap, norm in zip(soho_cube, cmaps, soho_norms.values()):
        cmap = get_cmap(cmap) if isinstance(cmap, str) else cmap
        s_map = s_map.rotate(recenter=True)
        bl = SkyCoord(-pos * u.arcsec, -pos * u.arcsec, frame=s_map.coordinate_frame)
        tr = SkyCoord(pos * u.arcsec, pos * u.arcsec, frame=s_map.coordinate_frame)
        s_map = s_map.submap(bl, top_right=tr)
        imsave(dir + '/soho_%d.jpg' % s_map.wavelength.value, cmap(norm(s_map.data))[..., :-1], check_contrast=False)
    for s_map, cmap, norm in zip(iti_cube, cmaps, [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304], sdo_norms['mag']]):
        cmap = get_cmap(cmap) if isinstance(cmap, str) else cmap
        s_map = s_map.rotate(recenter=True)
        bl = SkyCoord(-pos * u.arcsec, -pos * u.arcsec, frame=s_map.coordinate_frame)
        tr = SkyCoord(pos * u.arcsec, pos * u.arcsec, frame=s_map.coordinate_frame)
        s_map = s_map.submap(bl, top_right=tr)
        imsave(dir + '/iti_%d.jpg' % s_map.wavelength.value, cmap(norm(s_map.data))[..., :-1], check_contrast=False)
    for s_map, cmap, norm in zip(sdo_cube, cmaps, [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304], sdo_norms['mag']]):
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
        imsave(dir + '/sdo_%d.jpg' % s_map.wavelength.value, cmap(norm(s_map.data))[..., :-1], check_contrast=False)

# overview plot
# for soho_cube, iti_cube, sdo_cube, date in tqdm(zip(soho_maps, iti_maps, sdo_maps, dates)):
#     simplefilter('ignore')
#     fig, axs = plt.subplots(3, len(soho_cube), figsize=(3 * len(soho_cube), 9))
#     [ax.set_axis_off() for ax in np.ravel(axs)]
#     for ax, s_map, cmap, norm in zip(axs[0], soho_cube, cmaps, soho_norms.values()):
#         s_map = s_map.rotate(recenter=True)
#         bl = SkyCoord(-pos * u.arcsec, -pos * u.arcsec, frame=s_map.coordinate_frame)
#         tr = SkyCoord(pos * u.arcsec, pos * u.arcsec, frame=s_map.coordinate_frame)
#         s_map = s_map.submap(bl, top_right=tr)
#         s_map.plot(axes=ax, cmap=cmap, norm=norm, title=None)
#     for ax, s_map, cmap, norm in zip(axs[1], iti_cube, cmaps, [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304], sdo_norms['mag']]):
#         s_map = s_map.rotate(recenter=True)
#         bl = SkyCoord(-pos * u.arcsec, -pos * u.arcsec, frame=s_map.coordinate_frame)
#         tr = SkyCoord(pos * u.arcsec, pos * u.arcsec, frame=s_map.coordinate_frame)
#         s_map = s_map.submap(bl, top_right=tr)
#         s_map.plot(axes=ax, cmap=cmap, norm=norm, title=None)
#     for ax, s_map, cmap, norm in zip(axs[2], sdo_cube, cmaps, [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304], sdo_norms['mag']]):
#         if isinstance(s_map, AIAMap):
#             s_map = correct_degradation(s_map, correction_table=get_local_correction_table())
#             data = np.nan_to_num(s_map.data)
#             data = data / s_map.meta["exptime"]
#             s_map = Map(data, s_map.meta)
#         s_map = s_map.rotate(recenter=True)
#         bl = SkyCoord(-pos * u.arcsec, -pos * u.arcsec, frame=s_map.coordinate_frame)
#         tr = SkyCoord(pos * u.arcsec, pos * u.arcsec, frame=s_map.coordinate_frame)
#         s_map = s_map.submap(bl, top_right=tr)
#         s_map.plot(axes=ax, cmap=cmap, norm=norm, title=None)
#     plt.tight_layout()
#     fig.savefig(os.path.join(prediction_path, '%s.jpg') % date.isoformat())
#     plt.close(fig)
