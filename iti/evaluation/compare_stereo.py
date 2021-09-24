import glob
import os

from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from imageio import imsave
from sunpy.map import Map
from sunpy.visualization.colormaps import cm

from iti.data.editor import stereo_norms, sdo_norms

from astropy import units as u

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from iti.prediction.translate import STEREOToSDO

import numpy as np

# init
base_path = "/gss/r.jarolim/iti/stereo_v7"
prediction_path = os.path.join(base_path, 'evaluation')
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = STEREOToSDO(model_path=os.path.join(base_path, 'generator_AB.pt'))

cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
]

# translate
stereo_basenames = [
    [os.path.basename(f) for f in sorted(glob.glob('/gss/r.jarolim/data/stereo_iti2021_prep/%d/*.fits' % wl))] for wl in
    [171, 195, 284, 304]]
stereo_basenames = np.array(sorted(set(stereo_basenames[0]).intersection(*stereo_basenames[1:])))
stereo_dates = np.array([parse(bn.split('.')[0]) for bn in stereo_basenames])

cond = [(d.month == 11) or (d.month == 12) for d in stereo_dates]
stereo_basenames = stereo_basenames[cond]

stereo_maps = (
[Map('/gss/r.jarolim/data/stereo_iti2021_prep/%d/%s' % (wl, basename)) for wl in [171, 195, 284, 304]]
for basename in stereo_basenames)
iti_results = translator.translate('/gss/r.jarolim/data/stereo_iti2021_prep', basenames=stereo_basenames)

for stereo_cube, (iti_cube, _, _) in zip(stereo_maps, iti_results):
    date = iti_cube[0].date.datetime
    # solar_rotate_coordinate(ref_coord, observer=iti_cube[0])
    # iti_cube = [iti_map.submap(coord, width=width, height=height) for iti_map in iti_cube]
    # stereo_cube = [stereo_map.submap(coord, width=width, height=height) for stereo_map in stereo_cube]
    #
    print(iti_cube[0].date, stereo_cube[0].date)
    #
    path = os.path.join(prediction_path, date.isoformat('T'))
    if os.path.exists(path):
        continue
    os.makedirs(path, exist_ok=True)
    for s_map, cmap, norm in zip(stereo_cube, cmaps, stereo_norms.values()):
        s_map = s_map.rotate(recenter=True)
        bl = SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=s_map.coordinate_frame)
        tr = SkyCoord(1000 * u.arcsec, 1000 * u.arcsec, frame=s_map.coordinate_frame)
        s_map = s_map.submap(bl, tr)
        imsave(path + '/stereo_%d.jpg' % s_map.wavelength.value, cmap(norm(s_map.data))[..., :-1], check_contrast=False)

    for s_map, cmap, norm in zip(iti_cube, cmaps, list(sdo_norms.values())[2:6]):
        s_map = s_map.rotate(recenter=True)
        bl = SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=s_map.coordinate_frame)
        tr = SkyCoord(1000 * u.arcsec, 1000 * u.arcsec, frame=s_map.coordinate_frame)
        s_map = s_map.submap(bl, tr)
        imsave(path + '/iti_%d.jpg' % s_map.wavelength.value, cmap(norm(s_map.data))[..., :-1], check_contrast=False)
