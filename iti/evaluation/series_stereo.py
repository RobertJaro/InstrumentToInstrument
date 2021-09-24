import glob
import os
from datetime import datetime

from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import Helioprojective
from sunpy.map import Map
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.visualization.colormaps import cm
from astropy import units as u

from iti.data.editor import stereo_norms, sdo_norms

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from iti.prediction.translate import STEREOToSDO

from matplotlib import pyplot as plt

import numpy as np

# init
base_path = "/gss/r.jarolim/iti/stereo_v7"
prediction_path = os.path.join(base_path, 'series')
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
basenames = sorted([os.path.basename(f) for f in glob.glob('/gss/r.jarolim/data/stereo_iti2021_series_prep/171/*.fits')])[10:]
stereo_maps = ([Map('/gss/r.jarolim/data/stereo_iti2021_series_prep/%d/%s' % (wl, basename)) for wl in [171, 195, 284, 304]]
               for basename in basenames)

extend = 150
ref_coord = None
for stereo_cube, (iti_cube, _, _) in zip(stereo_maps, translator.translate('/gss/r.jarolim/data/stereo_iti2021_series_prep', basenames=basenames)):
    date = iti_cube[0].date.datetime
    #solar_rotate_coordinate(ref_coord, observer=iti_cube[0])
    #iti_cube = [iti_map.submap(coord, width=width, height=height) for iti_map in iti_cube]
    #stereo_cube = [stereo_map.submap(coord, width=width, height=height) for stereo_map in stereo_cube]
    #
    print(iti_cube[0].date, stereo_cube[0].date)
    if ref_coord is None:
        ref_coord = SkyCoord(70 * u.arcsec, -330 * u.arcsec, frame=iti_cube[0].coordinate_frame)
    coord = solar_rotate_coordinate(ref_coord, observer=iti_cube[0].observer_coordinate)
    #
    fig, axs = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle(date.isoformat(' ', timespec='minutes'), fontsize=18)
    for ax, s_map, cmap, norm in zip(axs[0], stereo_cube, cmaps, stereo_norms.values()):
        s_map.rotate(recenter=True).plot(axes=ax, cmap=cmap, norm=norm, title=None)
        ax.set_xlim(coord.Tx.value - extend, coord.Tx.value + extend)
        ax.set_ylim(coord.Ty.value - extend, coord.Ty.value + extend)
        ax.set_xlabel(None), ax.set_ylabel(None)
    for ax, s_map, cmap, norm in zip(axs[1], iti_cube, cmaps, list(sdo_norms.values())[2:6]):
        s_map.plot(axes=ax, cmap=cmap, norm=norm, title=None)
        ax.set_xlim(coord.Tx.value - extend, coord.Tx.value + extend)
        ax.set_ylim(coord.Ty.value - extend, coord.Ty.value + extend)
        ax.set_xlabel(None), ax.set_ylabel(None)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(prediction_path, '%s.jpg') % date.isoformat(), dpi=300)
    plt.close(fig)
