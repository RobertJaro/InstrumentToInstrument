import os

from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from sunpy.map import Map

from iti.translate import HMIToHinode



import numpy as np
from astropy import units as u

base_path = '/gpfs/gpfs0/robert.jarolim/iti/hmi_hinode_v4'
evaluation_path = os.path.join(base_path, 'plot')
os.makedirs(evaluation_path, exist_ok=True)

hmi_file = '/gpfs/gpfs0/robert.jarolim/data/iti/hmi_hinode_comparison/2013-11-18T17:46:20.fits'
hinode_file = '/gpfs/gpfs0/robert.jarolim/data/iti/hinode_iti2022_prep/FG20131118_174620.2.fits'

translator = HMIToHinode(model_path=os.path.join(base_path, 'generator_AB.pt'), patch_factor=3)
iti_map = next(translator.translate([hmi_file]))
iti_map = iti_map.rotate(missing=np.nan)
iti_map = iti_map.submap(bottom_left=SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=iti_map.coordinate_frame),
                         top_right=SkyCoord(1000 * u.arcsec, 1000 * u.arcsec, frame=iti_map.coordinate_frame))

hinode_map = Map(hinode_file)
# bl, tr = hinode_map.bottom_left_coord, hinode_map.top_right_coord
# hinode_map = hinode_map.rotate(scale=hinode_map.scale[0] / (0.15 * u.arcsec / u.pix), missing=np.nan)
# hinode_map = hinode_map.submap(bl, tr)

hmi_map = Map(hmi_file)
hmi_map = hmi_map.rotate(missing=np.nan)
hmi_map = hmi_map.submap(bottom_left=SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=hmi_map.coordinate_frame),
                         top_right=SkyCoord(1000 * u.arcsec, 1000 * u.arcsec, frame=hmi_map.coordinate_frame))


center = hinode_map.center
dim = hinode_map.top_right_coord.Tx - hinode_map.bottom_left_coord.Tx
bl = SkyCoord(center.Tx - dim / 2, center.Ty - dim/2, frame=hinode_map.coordinate_frame)
tr = SkyCoord(center.Tx + dim / 2, center.Ty + dim/2, frame=hinode_map.coordinate_frame)

hmi_sub_map = hmi_map.submap(bottom_left=bl, top_right=tr)
iti_sub_map = iti_map.submap(bottom_left=bl, top_right=tr)

plt.imsave(os.path.join(evaluation_path, 'hmi.jpg'), hmi_map.data, origin='lower', cmap='gray')
plt.imsave(os.path.join(evaluation_path, 'iti.jpg'), iti_map.data, origin='lower', cmap='gray')
plt.imsave(os.path.join(evaluation_path, 'hinode.jpg'), np.nan_to_num(hinode_map.data, np.nanmax(hinode_map.data)), origin='lower', cmap='gray')


plt.imsave(os.path.join(evaluation_path, 'hmi_sub.jpg'),hmi_sub_map.data, origin='lower', cmap='gray')
plt.imsave(os.path.join(evaluation_path, 'iti_sub.jpg'), iti_sub_map.data, origin='lower', cmap='gray')