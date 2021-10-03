import glob
import os
from random import sample

from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from sunpy.map import Map

from iti.translate import HMIToHinode

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
from skimage.io import imsave
from astropy import units as u

base_path = '/gss/r.jarolim/iti/hmi_hinode_v12'
evaluation_path = os.path.join(base_path, 'evaluation')
os.makedirs(evaluation_path, exist_ok=True)

hmi_files = glob.glob('/gss/r.jarolim/data/hmi_continuum/6173/*.fits')
hmi_files = [f for f in hmi_files if parse(os.path.basename(f).split('.')[0]).month in [11, 12]]
hmi_files = [f for f in hmi_files if '2014-12-22' in f]
hmi_files = sample(hmi_files, 1)

translator = HMIToHinode(model_path=os.path.join(base_path, 'generator_AB.pt'), patch_factor=3)
iti_map = next(translator.translate(hmi_files))
iti_map = iti_map.rotate(missing=np.nan)
iti_map = iti_map.submap(SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=iti_map.coordinate_frame),
                         SkyCoord(1000 * u.arcsec, 1000 * u.arcsec, frame=iti_map.coordinate_frame))

hmi_map = Map(hmi_files[0])
hmi_map = hmi_map.rotate(missing=np.nan)
hmi_map = hmi_map.submap(SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=hmi_map.coordinate_frame),
                         SkyCoord(1000 * u.arcsec, 1000 * u.arcsec, frame=hmi_map.coordinate_frame))

f_id = hmi_map.date.datetime.isoformat(' ', timespec='seconds')
imsave(os.path.join(evaluation_path, '%s_hmi.jpg' % f_id), np.flip(hmi_map.data, 0))
imsave(os.path.join(evaluation_path, '%s_iti.jpg' % f_id), np.flip(iti_map.data, 0))
