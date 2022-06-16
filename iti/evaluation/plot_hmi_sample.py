import glob
import os
from random import sample

from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from matplotlib import pyplot as plt
from sunpy.map import Map

from iti.translate import HMIToHinode



import numpy as np
from astropy import units as u

base_path = '/gpfs/gpfs0/robert.jarolim/iti/hmi_hinode_v1'
evaluation_path = os.path.join(base_path, 'evaluation')
os.makedirs(evaluation_path, exist_ok=True)

hmi_files = glob.glob('/gpfs/gpfs0/robert.jarolim/data/iti/hmi_continuum/*.fits')
hmi_files = [f for f in hmi_files if parse(os.path.basename(f).split('.')[0]).month in [11, 12]]
hmi_files = [f for f in hmi_files if '2014-12-22' in f]
hmi_files = sample(hmi_files, 1)

translator = HMIToHinode(model_path=os.path.join(base_path, 'generator_AB.pt'), patch_factor=3)
iti_map = next(translator.translate(hmi_files))
iti_map = iti_map.rotate(missing=np.nan)
iti_map = iti_map.submap(bottom_left=SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=iti_map.coordinate_frame),
                         top_right=SkyCoord(1000 * u.arcsec, 1000 * u.arcsec, frame=iti_map.coordinate_frame))

hmi_map = Map(hmi_files[0])
hmi_map = hmi_map.rotate(missing=np.nan)
hmi_map = hmi_map.submap(bottom_left=SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=hmi_map.coordinate_frame),
                         top_right=SkyCoord(1000 * u.arcsec, 1000 * u.arcsec, frame=hmi_map.coordinate_frame))

f_id = hmi_map.date.datetime.isoformat(' ', timespec='seconds')
plt.imsave(os.path.join(evaluation_path, '%s_hmi.jpg' % f_id), np.flip(hmi_map.data, 0), cmap='gray')
plt.imsave(os.path.join(evaluation_path, '%s_iti.jpg' % f_id), np.flip(iti_map.data, 0), cmap='gray')
