import glob
import os

from astropy.coordinates import SkyCoord
from skimage.metrics import structural_similarity

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from datetime import datetime

from dateutil.parser import parse
from sunpy.map import Map

from iti.data.align import alignMaps
from iti.download.download_hmi_continuum import HMIContinuumDownloader
from iti.prediction.translate import HMIToHinode
from matplotlib import pyplot as plt

import numpy as np
import  pandas as pd

from astropy import units as u

# Functions
base_path = '/gss/r.jarolim/iti/hmi_hinode_v12'
evaluation_path = os.path.join(base_path, "compare")
data_path = os.path.join(evaluation_path, "data")
os.makedirs(data_path, exist_ok=True)

df = pd.read_csv('/gss/r.jarolim/data/hinode/file_list.csv', index_col=False, parse_dates=['date'])
test_df = df[df.date.dt.month == 12]
test_df = test_df[test_df.classification == 'feature']

hinode_dates = [d.to_pydatetime() for d in test_df.date]
hinode_paths = test_df.file

fetcher = HMIContinuumDownloader(ds_path=data_path)
fetcher.fetchDates(hinode_dates)
hmi_paths = sorted(glob.glob(os.path.join(data_path, '6173/*.fits')))

translator = HMIToHinode(model_path=os.path.join(base_path, 'generator_AB.pt'), patch_factor=3)

# init maps generator
hinode_maps = (Map(path) for path in hinode_paths)
hmi_maps = (Map(path) for path in hmi_paths)

for hinode_path, hmi_map, iti_map, hinode_map in zip(hinode_paths, hmi_maps, translator.translate(hmi_paths), hinode_maps):
    scale = (0.15 * u.arcsec / u.pix)
    hinode_map = hinode_map.rotate(scale=hinode_map.scale[0] / scale, missing=np.mean(hinode_map.data))
    hmi_map = hmi_map.resample(u.Quantity(iti_map.dimensions, u.pix)).rotate(recenter=True)
    #
    alinged_hinode_map = alignMaps(hinode_map, hmi_map)
    bl = alinged_hinode_map.bottom_left_coord
    tr = alinged_hinode_map.top_right_coord
    #
    width, height = hinode_map.dimensions
    #
    hinode_data = hinode_map.data
    hmi_data = hmi_map.submap(bottom_left=bl, top_right=tr).resample(u.Quantity((width, height), u.pix)).data
    iti_data = iti_map.submap(bottom_left=bl, top_right=tr).resample(u.Quantity((width, height), u.pix)).data
    #
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    [ax.set_axis_off() for ax in np.ravel(axs)]
    axs[0, 0].imshow(hmi_data, cmap='gray')
    axs[0, 1].imshow(iti_data, cmap='gray')
    axs[0, 2].imshow(hinode_data, cmap='gray')
    # MSE
    square_error = (hmi_data - hinode_data) ** 2 / np.mean(hinode_data)
    axs[1, 0].imshow(square_error, vmin=0, vmax=1, )
    axs[1, 0].set_title('MSE %.03f' % np.mean(square_error))
    square_error = (iti_data - hinode_data) ** 2 / np.mean(hinode_data)
    axs[1, 1].imshow(square_error, vmin=0, vmax=1, )
    axs[1, 1].set_title('MSE %.03f' % np.mean(square_error))
    # SSIM
    ssim, ssim_img = structural_similarity(hmi_data, hinode_data, full=True)
    axs[2, 0].imshow(ssim_img, vmin=0, vmax=1, )
    axs[2, 0].set_title('SSIM %.03f' % ssim)
    ssim, ssim_img = structural_similarity(iti_data, hinode_data, full=True)
    axs[2, 1].imshow(ssim_img, vmin=0, vmax=1, )
    axs[2, 1].set_title('SSIM %.03f' % ssim)
    fig.tight_layout()
    plt.savefig(os.path.join(evaluation_path, os.path.basename(hinode_path).replace('.fits', '.jpg')), dpi=300)
    plt.close(fig)
