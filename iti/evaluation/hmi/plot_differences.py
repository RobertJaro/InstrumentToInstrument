import argparse
import glob
import os
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from imreg_dft import similarity, transform_img_dict
from matplotlib import pyplot as plt
from scipy.signal import correlate2d
from skimage import restoration
from skimage.metrics import structural_similarity
from skimage.transform import resize
from sunpy.map import Map
from tqdm import tqdm

from iti.data.editor import sdo_norms, hinode_norms
from iti.data.sdo.hmi_psf import load_psf
from iti.evaluation.compute_fid import calculate_fid_given_paths
from iti.evaluation.metrics import normalize, calibrate, ssim, psnr, rms_contrast, mae, rms_contrast_diff, \
    image_correlation
from iti.translate import HMIToHinode

parser = argparse.ArgumentParser(description='Plot differences between HMI and Hinode data.')
parser.add_argument('--out_path', type=str, help='Path to save evaluation results')
parser.add_argument('--model_path', type=str, help='Path to model file.')

args = parser.parse_args()




# Functions
evaluation_path = args.out_path
os.makedirs(evaluation_path, exist_ok=True)

hinode_paths = np.array(['/gpfs/gpfs0/robert.jarolim/data/iti/hinode_iti2022_prep/FG20141219_223311.7.fits'])
hmi_paths = np.array(['/gpfs/gpfs0/robert.jarolim/data/iti/hmi_hinode_comparison/2014-12-19T22:33:11.fits'])


translator = HMIToHinode(model_path=args.model_path)

# init maps generator
hinode_maps = (Map(path) for path in hinode_paths)
hmi_maps = (Map(path) for path in hmi_paths)

mean_hmi, std_hmi = (50392.45138124471, 9476.657264909856)
mean_hinode, std_hinode = (31149.955, 5273.3335)

hmi_norm = sdo_norms['continuum']
hinode_norm = hinode_norms['continuum']
psf = load_psf()

for i, (hmi_map, hinode_map, hinode_path) in tqdm(enumerate(zip(hmi_maps, hinode_maps, hinode_paths)),
                                                  total=len(hinode_paths)):
    # rescale, rotate, normalize and crop maps
    target_scale = (0.15 * u.arcsec / u.pix)
    scale_factor = hinode_map.scale[0] / target_scale
    new_dimensions = [int(hinode_map.data.shape[1] * scale_factor),
                      int(hinode_map.data.shape[0] * scale_factor)] * u.pixel
    hinode_map = hinode_map.resample(new_dimensions)
    hinode_map = Map(hinode_map.data.astype(np.float32), hinode_map.meta)
    hinode_center = hinode_map.center

    hmi_map = hmi_map.rotate(recenter=True, missing=0, order=4)
    scale_factor = hmi_map.scale[0].value / 0.6
    new_dimensions = [int(hmi_map.data.shape[1] * scale_factor),
                      int(hmi_map.data.shape[0] * scale_factor)] * u.pixel
    hmi_map = hmi_map.resample(new_dimensions)

    # crop Hinode data to 256x256
    crop = 256 # (min(hinode_map.data.shape) & -8) // 2 # find largest crop
    center_pix = hinode_map.world_to_pixel(SkyCoord(hinode_center.Tx, hinode_center.Ty, frame=hinode_map.coordinate_frame))
    c_y, c_x = int(np.ceil(center_pix.y.value)), int(np.ceil(center_pix.x.value))
    hinode_data = hinode_map.data[c_y - crop: c_y + crop, c_x - crop:c_x + crop]
    hinode_data = hinode_data / hinode_map.exposure_time.to(u.s).value
    # clip data
    hinode_data[hinode_data > 5e4] = 5e4
    hinode_data[hinode_data < 0] = 0

    # crop HMI data
    pad = ((crop // 4 + 7) & -8) - crop // 4  # find pix padding
    center_pix = hmi_map.world_to_pixel(SkyCoord(hinode_center.Tx, hinode_center.Ty, frame=hmi_map.coordinate_frame))
    c_y, c_x = int(center_pix.y.value), int(center_pix.x.value)
    hmi_data = hmi_map.data[c_y - (crop // 4 + pad): c_y + (crop // 4 + pad),
               c_x - (crop // 4 + pad):c_x + (crop // 4 + pad)]

    # translate ITI
    inp_tensor = torch.tensor(hmi_norm(hmi_data) * 2 - 1, dtype=torch.float32)[None, None]
    iti_data = translator.forward(inp_tensor)
    iti_data = hinode_norm.inverse((iti_data + 1) / 2)
    iti_data = iti_data[0, 0, pad * 4:-pad * 4, pad * 4:-pad * 4] if pad > 0 else iti_data[0, 0]

    # deconvolve
    hmi_data = (hmi_data - mean_hmi) / std_hmi * std_hinode + mean_hinode
    hmi_data = restoration.richardson_lucy(hmi_data, psf, clip=False)
    hmi_data = hmi_data[pad:-pad, pad:-pad] if pad > 0 else hmi_data
    # upsampling by 2
    hmi_data = resize(hmi_data, (crop * 2, crop * 2), order=3)

    normalized_iti_data = normalize(iti_data)
    normalized_hmi_data = normalize(hmi_data)
    normalized_hinode_data = normalize(hinode_data)
    try:
        transformation_iti = similarity(normalized_iti_data, normalized_hinode_data, numiter=20,
                                        constraints={'scale': (1, 0), 'angle': (0, 60)})
        transformation_hmi = similarity(normalized_hmi_data, normalized_hinode_data, numiter=20,
                                        constraints={'scale': (1, 0), 'angle': (0, 60)})
    except Exception as ex:
        print('ERROR', hinode_map.date.datetime.isoformat('T'))
        print(ex)
        continue

    # registrations of HMI are bad --> choose valid ITI registrations otherwise the dataset is too small
    hinode_registered_hmi = transform_img_dict(normalized_hinode_data, transformation_iti, bgval=0, order=3)
    hinode_registered_iti = transform_img_dict(normalized_hinode_data, transformation_iti, bgval=0, order=3)

    hmi_data, iti_data = normalized_hmi_data[80:-80, 80:-80], normalized_iti_data[80:-80, 80:-80]
    hinode_registered_hmi, hinode_registered_iti = hinode_registered_hmi[80:-80, 80:-80], hinode_registered_iti[80:-80, 80:-80]

    hmi_data = normalize(hmi_data)
    iti_data = normalize(iti_data)
    hinode_registered_hmi = normalize(hinode_registered_hmi)
    hinode_registered_iti = normalize(hinode_registered_iti)

    hmi_diff = np.abs(hmi_data - hinode_registered_hmi)
    iti_diff = np.abs(iti_data - hinode_registered_iti)

    vmax = np.nanmax(hmi_diff)
    p = os.path.join(evaluation_path, '%s_%s.jpg')
    plt.imsave(p % (os.path.basename(hinode_path), 'deconvolved'), hmi_data, cmap='gray', vmin=0, vmax=1)
    plt.imsave(p % (os.path.basename(hinode_path), 'iti'), iti_data, cmap='gray', vmin=0, vmax=1)
    plt.imsave(p % (os.path.basename(hinode_path), 'hinode'), hinode_registered_iti, cmap='gray', vmin=0, vmax=1)
    plt.imsave(p % (os.path.basename(hinode_path), 'diff_hmi'), hmi_diff, vmin=0, vmax=vmax, cmap='inferno')
    plt.imsave(p % (os.path.basename(hinode_path), 'diff_iti'), iti_diff, vmin=0, vmax=vmax, cmap='inferno')

    print('HMI DIFF', hmi_diff.mean())
    print('ITI DIFF', iti_diff.mean())

    # save colorbar
    im = plt.imshow(hmi_diff * 100, cmap='inferno', vmin=0, vmax=vmax * 100)
    fig = plt.figure(figsize=(4, 3))
    # with font size 28
    cbar = plt.colorbar(im)
    cbar.set_label(label='Absolute Difference [%]', size=14)
    plt.axis('off')
    plt.savefig(os.path.join(evaluation_path, '%s_%s.png') % (os.path.basename(hinode_path), 'colorbar'), dpi=300, transparent=True)
    plt.close(fig)

