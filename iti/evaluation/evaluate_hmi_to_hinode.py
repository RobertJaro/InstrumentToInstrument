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
from skimage import restoration
from skimage.metrics import structural_similarity
from skimage.transform import resize
from sunpy.map import Map
from tqdm import tqdm

from iti.data.editor import sdo_norms, hinode_norms
from iti.evaluation.compute_fid import calculate_fid_given_paths
from iti.translate import HMIToHinode

# Functions
base_path = '/gpfs/gpfs0/robert.jarolim/iti/hmi_hinode_v4'
evaluation_path = os.path.join(base_path, "evaluate_sample")
data_path = '/gpfs/gpfs0/robert.jarolim/data/iti/hmi_hinode_comparison'
os.makedirs(evaluation_path, exist_ok=True)
os.makedirs(os.path.join(evaluation_path, 'HMI'), exist_ok=True)
os.makedirs(os.path.join(evaluation_path, 'ITI'), exist_ok=True)
os.makedirs(os.path.join(evaluation_path, 'hinode'), exist_ok=True)

df = pd.read_csv('/gpfs/gpfs0/robert.jarolim/data/iti/hinode_file_list.csv', index_col=False, parse_dates=['date'])
test_df = df[df.date.dt.month.isin([11, 12])]
test_df = test_df[test_df.classification == 'feature']

invalid_matchings = [
    'FG20141107_191732.1.fits',
    'FG20141107_182559.4.fits',
    'FG20141116_182054.4.fits',
    'FG20141128_192240.0.fits',
    'FG20141217_010932.8.fits',
    'FG20141217_043222.2.fits',
    'FG20141219_094719.4.fits',
    'FG20151119_222310.7.fits',
    'FG20151217_225841.3.fits',
]

use_iti_registration = [
    'FG20111103_210316.2.fits',
    'FG20111104_230758.3.fits',
    'FG20111105_035102.4.fits',
    'FG20111106_100015.5.fits',
    'FG20131217_061942.6.fits',
    'FG20131222_151911.9.fits',
    'FG20131222_191158.4.fits',
    'FG20131223_155637.6.fits',
    'FG20141112_030441.2.fits',
    'FG20141115_115915.3.fits',
    'FG20141105_214929.4.fits',
    'FG20141106_050959.2.fits',
    'FG20141106_064650.8.fits',
    'FG20141107_040659.3.fits',
    'FG20141107_102701.3.fits',
    'FG20141107_120913.9.fits',
    'FG20141107_140052.0.fits',
    'FG20141107_150919.1.fits',
    'FG20141107_160027.8.fits',
    'FG20141110_124108.2.fits',
    'FG20141111_112619.0.fits',
    'FG20141129_143711.9.fits',
    'FG20141204_081323.1.fits',
    'FG20141205_120534.2.fits',
    'FG20141217_150449.6.fits',
    'FG20141218_165419.6.fits',
    'FG20141219_195749.0.fits',
]

exclude = invalid_matchings #+ use_iti_registration

hinode_paths = np.array([f for f in test_df.file if os.path.basename(f) not in exclude])
hinode_dates = [Map(f).date.datetime for f in hinode_paths]

hmi_paths = np.array(sorted(glob.glob(os.path.join(data_path, '*.fits'))))
hmi_dates = np.array([parse(os.path.basename(f).split('.')[0]) for f in hmi_paths])

hmi_paths = hmi_paths[[np.argmin(np.abs(hmi_dates - d)) for d in hinode_dates]]
hmi_dates = np.array([parse(os.path.basename(f).split('.')[0]) for f in hmi_paths])

cond = np.abs(hmi_dates - hinode_dates) < timedelta(seconds=10)
# cond = (np.abs(hmi_dates - hinode_dates) < timedelta(seconds=10)) & (np.abs(hmi_dates - datetime(2014, 12, 19, 22)) < timedelta(hours=2))
hmi_paths = hmi_paths[cond]
hinode_paths = hinode_paths[cond]

translator = HMIToHinode(model_path=os.path.join(base_path, 'generator_AB.pt'))

# init maps generator
hinode_maps = (Map(path) for path in hinode_paths)
hmi_maps = (Map(path) for path in hmi_paths)

mean_hmi, std_hmi = (50392.45138124471, 9476.657264909856)
mean_hinode, std_hinode = (31149.955, 5273.3335)


def calibrate_hmi(hmi_data):
    return (hmi_data - mean_hmi) * (std_hinode / std_hmi) + mean_hinode


def rms_contrast(data):
    return np.sqrt(np.nanmean((data - np.nanmean(data)) ** 2))

def psnr(img,img_ref):
    data_range = np.nanmax(img_ref) - np.nanmin(img_ref)
    mse = np.nanmean((img - img_ref) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)

def normalize(img):
    return (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))


coords = (np.stack(np.mgrid[:11, :11], 0) - 5) * 0.6
r = np.sqrt(coords[0] ** 2 + coords[1] ** 2)
phi = 2 * np.pi - np.arctan2(coords[0], coords[1])
c_w = [0.641, 0.211, 0.066, 0.00467, 0.035]
c_sig = [0.47, 1.155, 2.09, 4.42, 25.77]
c_a = [0.131, 0.371, 0.54, 0.781, 0.115]
c_u = [1, 1, 2, 1, 1]
c_nu = np.rad2deg([-1.85, 2.62, -2.34, 1.255, 2.58])
psf = np.sum([(1 + c_a[i] * np.cos(c_u[i] * phi + c_nu[i])) * c_w[i] * (
            1 / (2 * np.pi * c_sig[i] ** 2) * np.exp(-(r ** 2 / (2 * c_sig[i] ** 2)))) for i in range(5)], 0)

hmi_norm = sdo_norms['continuum']
hinode_norm = hinode_norms['continuum']

evaluation = {'iti_ssim': [], 'iti_psnr': [], 'iti_psnro': [], 'iti_rmsc': [], 'hmi_ssim': [], 'hmi_psnr': [], 'hmi_psnro': [], 'hmi_rmsc': []}
for i, (hmi_map, hinode_map, hinode_path) in tqdm(enumerate(zip(hmi_maps, hinode_maps, hinode_paths)),
                                                  total=len(hinode_paths)):
    # rescale, rotate, normalize and crop maps
    target_scale = (0.15 * u.arcsec / u.pix)
    scale_factor = hinode_map.scale[0] / target_scale
    new_dimensions = [int(hinode_map.data.shape[1] * scale_factor),
                      int(hinode_map.data.shape[0] * scale_factor)] * u.pixel
    hinode_map = hinode_map.resample(new_dimensions)
    hinode_map = Map(hinode_map.data.astype(np.float32), hinode_map.meta)
    coord = hinode_map.center

    hmi_map = hmi_map.rotate(recenter=True, missing=0, order=4)
    scale_factor = hmi_map.scale[0].value / 0.6
    new_dimensions = [int(hmi_map.data.shape[1] * scale_factor),
                      int(hmi_map.data.shape[0] * scale_factor)] * u.pixel
    hmi_map = hmi_map.resample(new_dimensions)

    crop = 256  # (min(hinode_map.data.shape) & -8) // 2 # find largest crop
    center_pix = hinode_map.world_to_pixel(SkyCoord(coord.Tx, coord.Ty, frame=hinode_map.coordinate_frame))
    c_y, c_x = int(np.ceil(center_pix.y.value)), int(np.ceil(center_pix.x.value))
    hinode_data = hinode_map.data[c_y - crop: c_y + crop, c_x - crop:c_x + crop]
    hinode_data = hinode_data / hinode_map.exposure_time.to(u.s).value
    # clip data
    hinode_data[hinode_data > 50000] = 50000
    hinode_data[hinode_data < 0] = 0

    pad = ((crop // 4 + 7) & -8) - crop // 4  # find pix padding
    center_pix = hmi_map.world_to_pixel(SkyCoord(coord.Tx, coord.Ty, frame=hmi_map.coordinate_frame))
    c_y, c_x = int(center_pix.y.value), int(center_pix.x.value)
    hmi_data = hmi_map.data[c_y - (crop // 4 + pad): c_y + (crop // 4 + pad),
               c_x - (crop // 4 + pad):c_x + (crop // 4 + pad)]
    # translate ITI
    inp_tensor = torch.tensor(hmi_norm(hmi_data) * 2 - 1, dtype=torch.float32)[None, None].cuda()
    iti_data = translator.generator(inp_tensor)
    iti_data = hinode_norm.inverse((iti_data[0].detach().cpu().numpy() + 1) / 2)
    iti_data = iti_data[0, pad * 4:-pad * 4, pad * 4:-pad * 4] if pad > 0 else iti_data[0]
    # save original
    original_hmi_data = hmi_data.copy()[pad:-pad, pad:-pad] if pad > 0 else hmi_data.copy()
    original_hmi_data = calibrate_hmi(original_hmi_data)
    # deconvolve
    hmi_data = restoration.richardson_lucy(hmi_data, psf, iterations=30, clip=False)
    hmi_data = hmi_data[pad:-pad, pad:-pad] if pad > 0 else hmi_data
    # upsampling by 4
    hmi_data = resize(hmi_data, (crop * 2, crop * 2), order=3)
    # calibrate HMI images
    hmi_data = calibrate_hmi(hmi_data)

    normalized_iti_data = normalize(iti_data)
    normalized_hmi_data = normalize(hmi_data)
    normalized_hinode_data = normalize(hinode_data)
    try:
        transformation_iti = similarity(normalized_iti_data, normalized_hinode_data, numiter=20,
                                        constraints={'scale': (1, 0), 'angle': (0, 60)})
        # transformation_hmi = similarity(normalized_hmi_data, normalized_hinode_data, numiter=20,
        #                                 constraints={'scale': (1, 0), 'angle': (0, 60)})
    except Exception as ex:
        print('ERROR', hinode_map.date.datetime.isoformat('T'))
        print(ex)
        continue
    # if os.path.basename(hinode_path) in use_iti_registration:
    #     transformation_hmi = transformation_iti
    transformation_hmi = transformation_iti

    registered_iti = transform_img_dict(hinode_data, transformation_iti, bgval=0, order=3)
    registered_hmi = transform_img_dict(hinode_data, transformation_hmi, bgval=0, order=3)

    registered_hmi, registered_iti = registered_hmi[80:-80, 80:-80], registered_iti[80:-80, 80:-80]
    hmi_data, iti_data = hmi_data[80:-80, 80:-80], iti_data[80:-80, 80:-80]
    original_hmi_data = original_hmi_data[20:-20, 20:-20]
    #
    normalized_iti_data = normalize(iti_data)
    normalized_hmi_data = normalize(hmi_data)
    normalized_registered_iti = normalize(registered_iti)
    normalized_registered_hmi = normalize(registered_hmi)

    # vmax = np.nanmax(np.abs(normalized_hmi_data - normalized_registered_hmi))
    # p = os.path.join(evaluation_path, '%s_%s.jpg')
    # plt.imsave(p % (os.path.basename(hinode_path), 'original'), original_hmi_data, cmap='gray')
    # plt.imsave(p % (os.path.basename(hinode_path), 'deconvolved'), hmi_data, cmap='gray')
    # plt.imsave(p % (os.path.basename(hinode_path), 'iti'), iti_data, cmap='gray')
    # plt.imsave(p % (os.path.basename(hinode_path), 'hinode'), registered_iti, cmap='gray')
    # plt.imsave(p % (os.path.basename(hinode_path), 'diff_original'), np.abs(normalized_hmi_data - normalized_registered_hmi), vmin=0, vmax=vmax, cmap='inferno')
    # plt.imsave(p % (os.path.basename(hinode_path), 'diff_iti'), np.abs(normalized_iti_data - normalized_registered_iti), vmin=0, vmax=vmax, cmap='inferno')

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(original_hmi_data, cmap='gray', extent=(0, original_hmi_data.shape[0] * 0.6, 0, original_hmi_data.shape[1] * 0.6))
    ax.set_xlabel('X [arcsec]')
    ax.set_ylabel('Y [arcsec]')
    fig.tight_layout()
    fig.savefig(os.path.join(evaluation_path, '%s_coord.jpg' % os.path.basename(hinode_path)))
    plt.close(fig)

    vmax = np.nanmax(np.abs(normalized_hmi_data - normalized_registered_hmi))
    fig, axs = plt.subplots(1, 6, figsize=(21, 4))
    [(ax.set_xticks([]), ax.set_yticks([])) for ax in axs]
    axs[0].imshow(original_hmi_data, cmap='gray',)# vmin=0, vmax=50000)
    axs[1].imshow(hmi_data, cmap='gray', )#vmin=0, vmax=50000)
    axs[2].imshow(iti_data, cmap='gray', )#vmin=0, vmax=50000)
    axs[3].imshow(registered_iti, cmap='gray', )#vmin=0, vmax=50000)
    axs[4].imshow(np.abs(normalized_hmi_data - normalized_registered_hmi), vmin=0, vmax=vmax)
    axs[5].imshow(np.abs(normalized_iti_data - normalized_registered_iti), vmin=0, vmax=vmax)
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(evaluation_path, '%s.jpg' % os.path.basename(hinode_path)))
    plt.close(fig)

    data_range = np.nanmax(normalized_registered_iti) - np.nanmin(normalized_registered_iti) # Hinode data range
    hmi_mse = np.nanmean((normalized_hmi_data - normalized_registered_hmi) ** 2)
    hmi_psnr = 10 * np.log10((data_range ** 2) / hmi_mse)
    iti_mse = np.nanmean((normalized_iti_data - normalized_registered_iti) ** 2)
    iti_psnr = 10 * np.log10((data_range ** 2) / iti_mse)

    hmi_psnro = psnr(hmi_data, registered_hmi)
    iti_psnro = psnr(iti_data, registered_iti)

    hinode_rmsc, hmi_rmsc, iti_rmsc = rms_contrast(normalized_registered_iti), rms_contrast(normalized_hmi_data), rms_contrast(normalized_iti_data)

    hmi_ssim = structural_similarity(normalized_hmi_data, normalized_registered_hmi, data_range=data_range)
    iti_ssim = structural_similarity(normalized_iti_data, normalized_registered_iti, data_range=data_range)

    plt.imsave(os.path.join(evaluation_path, 'ITI', '%s.jpg' % hinode_map.date.datetime.isoformat('T')), iti_data,
               cmap='gray',)# vmin=0, vmax=50000)
    plt.imsave(os.path.join(evaluation_path, 'hinode', '%s.jpg' % hinode_map.date.datetime.isoformat('T')),
               registered_iti,
               cmap='gray',)# vmin=0, vmax=50000)
    plt.imsave(os.path.join(evaluation_path, 'HMI', '%s.jpg' % hinode_map.date.datetime.isoformat('T')), hmi_data,
               cmap='gray',)# vmin=0, vmax=50000)

    print('RESULT (HMI, ITI)', hinode_map.date.datetime.isoformat('T'))
    print('PSNR (original)', hmi_psnro, iti_psnro)
    print('MSE', hmi_mse, iti_mse)
    print('PSNR', hmi_psnr, iti_psnr)
    print('SSIM', hmi_ssim, iti_ssim)
    print('RMS contrast', hinode_rmsc, hmi_rmsc, iti_rmsc)
    print('STAT MEAN', np.nanmean(hmi_data) / np.nanmean(hinode_data), np.nanmean(iti_data) / np.nanmean(hinode_data), )
    print('STAT STD', np.nanstd(hmi_data) / np.nanstd(hinode_data), np.nanstd(iti_data) / np.nanstd(hinode_data), )
    print('SUCCESS', transformation_hmi['success'], transformation_iti['success'])

    evaluation['iti_ssim'] += [iti_ssim]
    evaluation['iti_psnr'] += [iti_psnr]
    evaluation['iti_psnro'] += [iti_psnro]
    evaluation['iti_rmsc'] += [iti_rmsc]
    evaluation['hmi_ssim'] += [hmi_ssim]
    evaluation['hmi_psnr'] += [hmi_psnr]
    evaluation['hmi_psnro'] += [hmi_psnro]
    evaluation['hmi_rmsc'] += [hmi_rmsc]

fid_iti = calculate_fid_given_paths((os.path.join(evaluation_path, 'hinode'), os.path.join(evaluation_path, 'ITI')), 4,
                                    'cuda', 2048)
fid_hmi = calculate_fid_given_paths((os.path.join(evaluation_path, 'hinode'), os.path.join(evaluation_path, 'HMI')), 4,
                                    'cuda', 2048)

with open(os.path.join(evaluation_path, 'evaluation.txt'), 'w') as f:
    print('(ssim, psnr, psnr-o, rmsc, FID)', file=f)
    print('ITI', file=f)
    print('(%f, %f, %f, %f, %f)' % (
    np.mean(evaluation['iti_ssim']), np.mean(evaluation['iti_psnr']), np.mean(evaluation['iti_psnro']), np.mean(evaluation['iti_rmsc']), fid_iti), file=f)
    print('HMI', file=f)
    print('(%f, %f, %f, %f, %f)' % (
    np.mean(evaluation['hmi_ssim']), np.mean(evaluation['hmi_psnr']), np.mean(evaluation['hmi_psnro']), np.mean(evaluation['hmi_rmsc']), fid_hmi), file=f)
