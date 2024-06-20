import glob
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from aiapy.calibrate import register
from astropy import units as u
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from imreg_dft import similarity, transform_img_dict
from sunpy.map import Map
from tqdm import tqdm

# Functions
base_path = '/gpfs/gpfs0/robert.jarolim/iti/hmi_hinode_baseline'
os.makedirs(base_path, exist_ok=True)

data_path = '/gpfs/gpfs0/robert.jarolim/data/iti/hmi_hinode_baseline'

df = pd.read_csv('/gpfs/gpfs0/robert.jarolim/data/iti/hinode_file_list.csv', index_col=False, parse_dates=['date'])
test_df = df[df.date.dt.month.isin(list(range(2, 10)))]
test_df = test_df[test_df.classification == 'feature']

hinode_paths = test_df.file
hinode_dates = [Map(f).date.datetime for f in hinode_paths]

hmi_paths = np.array(sorted(glob.glob(os.path.join(data_path, '*.fits'))))
hmi_dates = np.array([parse(os.path.basename(f).split('.')[0]) for f in hmi_paths])

hmi_paths = hmi_paths[[np.argmin(np.abs(hmi_dates - d)) for d in hinode_dates]]
hmi_dates = np.array([parse(os.path.basename(f).split('.')[0]) for f in hmi_paths])

cond = (np.abs(hmi_dates - hinode_dates) < timedelta(seconds=10))
hmi_paths = hmi_paths[cond]
hinode_paths = hinode_paths[cond]

# init maps generator
hinode_maps = (Map(path) for path in hinode_paths)
hmi_maps = (Map(path) for path in hmi_paths)

hmi_distribution = []
hinode_distribution = []

for hmi_map, hinode_map in tqdm(zip(hmi_maps, hinode_maps), total=len(hinode_paths)):
    # rescale, rotate, normalize and crop hinode map
    target_scale = (0.15 * u.arcsec / u.pix)
    hinode_map = hinode_map.rotate(scale=hinode_map.scale[0] / target_scale, missing=np.nan)
    hmi_map = register(hmi_map)
    coord = hinode_map.center

    hinode_patch = hinode_map.submap(
        bottom_left=SkyCoord(coord.Tx - 50 * u.arcsec, coord.Ty - 50 * u.arcsec, frame=hinode_map.coordinate_frame),
        top_right=SkyCoord(coord.Tx + 50 * u.arcsec, coord.Ty + 50 * u.arcsec, frame=hinode_map.coordinate_frame))
    hinode_data = hinode_patch.data / hinode_patch.exposure_time.to(u.s).value
    # remove overlap for invalid submap (additional pix)
    crop_dim = np.min(hinode_data.shape)
    hinode_data = hinode_data[:crop_dim, :crop_dim]

    hmi_patch = hmi_map.submap(
        bottom_left=SkyCoord(coord.Tx - 50 * u.arcsec, coord.Ty - 50 * u.arcsec, frame=hmi_map.coordinate_frame),
        top_right=SkyCoord(coord.Tx + 50 * u.arcsec, coord.Ty + 50 * u.arcsec, frame=hmi_map.coordinate_frame))
    hmi_patch = hmi_patch.resample(hinode_data.shape * u.pix)
    hmi_data = hmi_patch.data

    normalized_hinode_data = (hinode_data - np.median(hinode_data)) / np.std(hinode_data)
    normalized_hmi_data = (hmi_data - np.median(hmi_data)) / np.std(hmi_data)
    try:
        transformation = similarity(normalized_hmi_data, normalized_hinode_data, numiter=5)
    except Exception as ex:
        print('ERROR', hinode_map.date.datetime.isoformat('T'))
        print(ex)
        continue
    if transformation['success'] < 0.09:
        print('Not aligned; SUCCESS =', transformation['success'])
        continue

    registered_data = transform_img_dict(hinode_data, transformation, bgval=np.NAN)

    hmi_data[np.isnan(registered_data)] = np.nan
    hmi_distribution += [hmi_data]
    hinode_distribution += [registered_data]

hmi_res = (np.nanmean(hmi_distribution), np.nanstd(hmi_distribution))
hinode_res = (np.nanmean(hinode_distribution), np.nanstd(hinode_distribution))

with open(os.path.join(base_path, 'calibration.txt'), 'w') as f:
    print('HMI (mean, std)', file=f)
    print('(%f, %f)' % hmi_res, file=f)
    print('Hinode (mean, std)', file=f)
    print('(%f, %f)' % hinode_res, file=f)
