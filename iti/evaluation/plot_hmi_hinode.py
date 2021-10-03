import os

from astropy.coordinates import SkyCoord
from sunpy.map import Map

from iti.translate import HMIToHinode

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
from skimage.io import imsave
from astropy import units as u

base_path = '/gss/r.jarolim/iti/hmi_hinode_v12'
evaluation_path = os.path.join(base_path, 'evaluation')
os.makedirs(evaluation_path, exist_ok=True)

hmi_file = '/gss/r.jarolim/data/hmi_hinode_comparison/6173/2013-11-18T17:46:20.fits'
hinode_file = '/gss/r.jarolim/data/hinode/level1/FG20131118_174620.2.fits'

translator = HMIToHinode(model_path=os.path.join(base_path, 'generator_AB.pt'), patch_factor=3)
iti_map = next(translator.translate([hmi_file]))
iti_map = iti_map.rotate(missing=np.nan)
iti_map = iti_map.submap(SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=iti_map.coordinate_frame),
                         SkyCoord(1000 * u.arcsec, 1000 * u.arcsec, frame=iti_map.coordinate_frame))

hinode_map = Map(hinode_file)
# bl, tr = hinode_map.bottom_left_coord, hinode_map.top_right_coord
# hinode_map = hinode_map.rotate(scale=hinode_map.scale[0] / (0.15 * u.arcsec / u.pix), missing=np.nan)
# hinode_map = hinode_map.submap(bl, tr)

hmi_map = Map(hmi_file)
hmi_map = hmi_map.rotate(missing=np.nan)
hmi_map = hmi_map.submap(SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=hmi_map.coordinate_frame),
                         SkyCoord(1000 * u.arcsec, 1000 * u.arcsec, frame=hmi_map.coordinate_frame))


center = hinode_map.center
dim = hinode_map.top_right_coord.Tx - hinode_map.bottom_left_coord.Tx
bl = SkyCoord(center.Tx - dim / 2, center.Ty - dim/2, frame=hinode_map.coordinate_frame)
tr = SkyCoord(center.Tx + dim / 2, center.Ty + dim/2, frame=hinode_map.coordinate_frame)

hmi_sub_map = hmi_map.submap(bl, tr)
iti_sub_map = iti_map.submap(bl, tr)

imsave(os.path.join(evaluation_path, 'hmi.jpg'), np.flip(hmi_map.data, 0))
imsave(os.path.join(evaluation_path, 'iti.jpg'), np.flip(iti_map.data, 0))
imsave(os.path.join(evaluation_path, 'hinode.jpg'), np.flip(np.nan_to_num(hinode_map.data, np.nanmax(hinode_map.data)), 0))


imsave(os.path.join(evaluation_path, 'hmi_sub.jpg'), np.flip(hmi_sub_map.data, 0))
imsave(os.path.join(evaluation_path, 'iti_sub.jpg'), np.flip(iti_sub_map.data, 0))