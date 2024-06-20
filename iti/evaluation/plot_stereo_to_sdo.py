import glob
import os

from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from imageio import imsave
from sunpy.map import Map
from sunpy.visualization.colormaps import cm

from iti.data.dataset import get_intersecting_files
from iti.data.editor import stereo_norms, sdo_norms

from astropy import units as u



from iti.translate import STEREOToSDO

import numpy as np

# init
base_path = "/gpfs/gpfs0/robert.jarolim/iti/stereo_to_sdo_v1"
prediction_path = os.path.join(base_path, 'evaluation')
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = STEREOToSDO(model_path=os.path.join(base_path, 'generator_AB.pt'), n_workers=1)

cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
]

# translate
stereo_files = get_intersecting_files('/gpfs/gpfs0/robert.jarolim/data/iti/stereo_iti2021_prep', ['171', '195', '284', '304', ],
                                      ext='.fits', months=[11, 12], years=[2008])

iti_results = translator.translate(stereo_files)

stereo_files = np.array(stereo_files)
stereo_maps = ([Map(f) for f in stereo_files[:, i]] for i in range(stereo_files.shape[1]))

for stereo_cube, iti_cube in zip(stereo_maps, iti_results):
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
        s_map = s_map.submap(bottom_left=bl, top_right=tr)
        imsave(path + '/stereo_%d.jpg' % s_map.wavelength.value, cmap(norm(s_map.data))[..., :-1], check_contrast=False)

    for s_map, cmap, norm in zip(iti_cube, cmaps, list(sdo_norms.values())[2:6]):
        s_map = s_map.rotate(recenter=True)
        bl = SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=s_map.coordinate_frame)
        tr = SkyCoord(1000 * u.arcsec, 1000 * u.arcsec, frame=s_map.coordinate_frame)
        s_map = s_map.submap(bottom_left=bl, top_right=tr)
        imsave(path + '/iti_%d.jpg' % s_map.wavelength.value, cmap(norm(s_map.data))[..., :-1], check_contrast=False)
