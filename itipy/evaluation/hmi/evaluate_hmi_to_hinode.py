import argparse
import glob
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from imreg_dft import similarity, transform_img_dict
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import restoration
from skimage.transform import resize
from sunpy.map import Map
from tqdm import tqdm

from itipy.data.editor import sdo_norms, hinode_norms
from itipy.data.sdo.hmi_psf import load_psf
from itipy.evaluation.compute_fid import calculate_fid_given_paths
from itipy.evaluation.metrics import normalize, ssim, psnr, mae, rms_contrast_diff, \
    image_correlation
from itipy.translate import HMIToHinode

parser = argparse.ArgumentParser(description='Evaluate paired samples of HMI to Hinode translation')
parser.add_argument('--out_path', type=str, help='Path to save evaluation results')
parser.add_argument('--hinode_data', type=str, help='Path to Hinode CSV file.')
parser.add_argument('--hmi_data', type=str, help='Path to HMI data directory.')
parser.add_argument('--model_path', type=str, help='Path to model file.')

args = parser.parse_args()

# Functions
evaluation_path = args.out_path

hmi_evaluation_path = os.path.join(evaluation_path, 'HMI')
iti_evaluation_path = os.path.join(evaluation_path, 'ITI')
hinode_evaluation_path = os.path.join(evaluation_path, 'hinode')

os.makedirs(evaluation_path, exist_ok=True)
os.makedirs(hmi_evaluation_path, exist_ok=True)
os.makedirs(iti_evaluation_path, exist_ok=True)
os.makedirs(hinode_evaluation_path, exist_ok=True)

df = pd.read_csv(args.hinode_data, index_col=False, parse_dates=['date'])
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
    # new entries
    'FG20141107_102701.3.fits_coord.jpg',

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

exclude = invalid_matchings  # + use_iti_registration

hinode_paths = np.array([f for f in test_df.file if os.path.basename(f) not in exclude])
hinode_dates = [Map(f).date.datetime for f in hinode_paths]

hmi_paths = np.array(sorted(glob.glob(os.path.join(args.hmi_data, '*.fits'))))
hmi_dates = np.array([parse(os.path.basename(f).split('.')[0]) for f in hmi_paths])

hmi_paths = hmi_paths[[np.argmin(np.abs(hmi_dates - d)) for d in hinode_dates]]
hmi_dates = np.array([parse(os.path.basename(f).split('.')[0]) for f in hmi_paths])

cond = np.abs(hmi_dates - hinode_dates) < timedelta(seconds=10)

hmi_paths = hmi_paths[cond]
hinode_paths = hinode_paths[cond]

translator = HMIToHinode(model_path=args.model_path)

# init maps generator
hinode_maps = (Map(path) for path in hinode_paths)
hmi_maps = (Map(path) for path in hmi_paths)

mean_hmi, std_hmi = (44843.92, 9413.19)  # (50392.45138124471, 9476.657264909856)
mean_hinode, std_hinode = (31149.955, 5273.3335)

hmi_norm = sdo_norms['continuum']
hinode_norm = hinode_norms['continuum']
psf = load_psf()

evaluation = {
    'iti': {'ssim': [], 'psnr': [], 'mae': [], 'rmsc': [], 'cc': []},
    'hmi': {'ssim': [], 'psnr': [], 'mae': [], 'rmsc': [], 'cc': []}
}

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
    crop = 256  # (min(hinode_map.data.shape) & -8) // 2 # find largest crop
    center_pix = hinode_map.world_to_pixel(
        SkyCoord(hinode_center.Tx, hinode_center.Ty, frame=hinode_map.coordinate_frame))
    c_y, c_x = int(np.ceil(center_pix.y.value)), int(np.ceil(center_pix.x.value))
    hinode_data = hinode_map.data[c_y - crop: c_y + crop, c_x - crop:c_x + crop]
    hinode_data = hinode_data / hinode_map.exposure_time.to(u.s).value
    # clip data
    hinode_data[hinode_data > 5e4] = 5e4
    hinode_data[hinode_data < 0] = 0

    # crop HMI data to 256x256 + padding
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
    original_hmi_data = hmi_data.copy()
    hmi_data = restoration.richardson_lucy(hmi_data, psf, clip=False)
    hmi_data = hmi_data[pad:-pad, pad:-pad] if pad > 0 else hmi_data
    original_hmi_data = original_hmi_data[pad:-pad, pad:-pad] if pad > 0 else original_hmi_data
    # upsampling by 2
    hmi_data = resize(hmi_data, (crop * 2, crop * 2), order=3)
    original_hmi_data = resize(original_hmi_data, (crop * 2, crop * 2), order=3)

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
    hinode_registered_hmi = transform_img_dict(hinode_data, transformation_iti, bgval=0, order=3)
    hinode_registered_iti = transform_img_dict(hinode_data, transformation_iti, bgval=0, order=3)

    hmi_data, iti_data = hmi_data[80:-80, 80:-80], iti_data[80:-80, 80:-80]
    hinode_registered_hmi, hinode_registered_iti = hinode_registered_hmi[80:-80, 80:-80], hinode_registered_iti[80:-80,
                                                                                          80:-80]
    original_hmi_data = original_hmi_data[80:-80, 80:-80]

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    extent = (0, hinode_data.shape[0] * 0.15, 0, hinode_data.shape[1] * 0.15)
    axs[0].imshow(hinode_registered_iti, cmap='gray', vmin=0, vmax=5e4, extent=extent)
    axs[0].set_title('Hinode - ITI registered')
    axs[1].imshow(hinode_registered_hmi, cmap='gray', vmin=0, vmax=5e4, extent=extent)
    axs[1].set_title('Hinode - HMI registered')
    axs[2].imshow(hmi_data, cmap='gray', vmin=0, vmax=5e4, extent=extent)
    axs[2].set_title('HMI')
    axs[3].imshow(iti_data, cmap='gray', vmin=0, vmax=5e4, extent=extent)
    axs[3].set_title('ITI')
    [ax.set_xlabel('X [arcsec]') for ax in axs]
    [ax.set_ylabel('Y [arcsec]') for ax in axs]
    fig.tight_layout()
    fig.savefig(os.path.join(evaluation_path, '%s_coord.jpg' % os.path.basename(hinode_path)))
    plt.close(fig)

    hmi_diff = np.abs(hmi_data - hinode_registered_hmi) / 5e4 * 100
    iti_diff = np.abs(iti_data - hinode_registered_iti) / 5e4 * 100

    fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    extent = (0, hinode_data.shape[0] * 0.15, 0, hinode_data.shape[1] * 0.15)
    im = axs[0].imshow(original_hmi_data, cmap='gray', vmin=0, vmax=5e4, extent=extent)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal', label='HMI [DN/s]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cbar.set_ticks([0, 2e4, 4e4])
    cbar.set_ticklabels(['0', '2e4', '4e4'])
    #
    im = axs[1].imshow(hmi_data, cmap='gray', vmin=0, vmax=5e4, extent=extent)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal', label='Deconvolved [DN/s]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cbar.set_ticks([0, 2e4, 4e4])
    cbar.set_ticklabels(['0', '2e4', '4e4'])
    #
    im = axs[2].imshow(iti_data, cmap='gray', vmin=0, vmax=5e4, extent=extent)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal', label='ITI [DN/s]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cbar.set_ticks([0, 2e4, 4e4])
    cbar.set_ticklabels(['0', '2e4', '4e4'])
    #
    im = axs[3].imshow(hinode_registered_iti, cmap='gray', vmin=0, vmax=5e4, extent=extent)
    divider = make_axes_locatable(axs[3])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal', label='Hinode/BFI [DN/s]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cbar.set_ticks([0, 2e4, 4e4])
    cbar.set_ticklabels(['0', '2e4', '4e4'])
    #
    im = axs[4].imshow(hmi_diff, cmap='inferno', vmin=0, vmax=40, extent=extent)
    divider = make_axes_locatable(axs[4])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal', label='$\Delta$ Deconv. [%]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    #
    im = axs[5].imshow(iti_diff, cmap='inferno', vmin=0, vmax=40, extent=extent)
    divider = make_axes_locatable(axs[5])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal', label='$\Delta$ ITI [%]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    #
    [ax.set_xlabel('X [arcsec]') for ax in axs]
    axs[0].set_ylabel('Y [arcsec]')
    fig.tight_layout()
    fig.savefig(os.path.join(evaluation_path, '%s_res.png' % os.path.basename(hinode_path)),
                dpi=300, transparent=True)
    plt.close(fig)

    # vmax = np.nanmax(np.abs(normalized_hmi_data - normalized_registered_hmi))
    # p = os.path.join(evaluation_path, '%s_%s.jpg')
    # plt.imsave(p % (os.path.basename(hinode_path), 'original'), original_hmi_data, cmap='gray')
    # plt.imsave(p % (os.path.basename(hinode_path), 'deconvolved'), hmi_data, cmap='gray')
    # plt.imsave(p % (os.path.basename(hinode_path), 'iti'), iti_data, cmap='gray')
    # plt.imsave(p % (os.path.basename(hinode_path), 'hinode'), registered_iti, cmap='gray')
    # plt.imsave(p % (os.path.basename(hinode_path), 'diff_original'), np.abs(normalized_hmi_data - normalized_registered_hmi), vmin=0, vmax=vmax, cmap='inferno')
    # plt.imsave(p % (os.path.basename(hinode_path), 'diff_iti'), np.abs(normalized_iti_data - normalized_registered_iti), vmin=0, vmax=vmax, cmap='inferno')

    # vmax = np.nanmax(np.abs(normalized_hmi_data - normalized_registered_hmi))
    # fig, axs = plt.subplots(1, 6, figsize=(21, 4))
    # [(ax.set_xticks([]), ax.set_yticks([])) for ax in axs]
    # axs[0].imshow(original_hmi_data, cmap='gray',)# vmin=0, vmax=50000)
    # axs[1].imshow(hmi_data, cmap='gray', )#vmin=0, vmax=50000)
    # axs[2].imshow(iti_data, cmap='gray', )#vmin=0, vmax=50000)
    # axs[3].imshow(registered_iti, cmap='gray', )#vmin=0, vmax=50000)
    # axs[4].imshow(np.abs(normalized_hmi_data - normalized_registered_hmi), vmin=0, vmax=vmax)
    # axs[5].imshow(np.abs(normalized_iti_data - normalized_registered_iti), vmin=0, vmax=vmax)
    # fig.tight_layout(pad=0)
    # fig.savefig(os.path.join(evaluation_path, '%s.jpg' % os.path.basename(hinode_path)))
    # plt.close(fig)
    #
    # data_range = np.nanmax(normalized_registered_iti) - np.nanmin(normalized_registered_iti) # Hinode data range
    # hmi_mse = np.nanmean((normalized_hmi_data - normalized_registered_hmi) ** 2)
    # hmi_psnr = 10 * np.log10((data_range ** 2) / hmi_mse)
    # iti_mse = np.nanmean((normalized_iti_data - normalized_registered_iti) ** 2)
    # iti_psnr = 10 * np.log10((data_range ** 2) / iti_mse)
    #
    # hmi_psnro = psnr(hmi_data, registered_hmi)
    # iti_psnro = psnr(iti_data, registered_iti)
    #
    # hinode_rmsc, hmi_rmsc, iti_rmsc = rms_contrast(normalized_registered_iti), rms_contrast(normalized_hmi_data), rms_contrast(normalized_iti_data)
    #
    #
    # iti_ssim = structural_similarity(normalized_iti_data, normalized_registered_iti, data_range=data_range)
    #
    plt.imsave(os.path.join(evaluation_path, 'ITI', '%s.jpg' % hinode_map.date.datetime.isoformat('T')), iti_data,
               cmap='gray', vmin=0, vmax=50000)
    plt.imsave(os.path.join(evaluation_path, 'hinode', '%s.jpg' % hinode_map.date.datetime.isoformat('T')),
               hinode_registered_iti, cmap='gray', vmin=0, vmax=50000)
    plt.imsave(os.path.join(evaluation_path, 'HMI', '%s.jpg' % hinode_map.date.datetime.isoformat('T')), hmi_data,
               cmap='gray', vmin=0, vmax=50000)
    #
    # print('RESULT (HMI, ITI)', hinode_map.date.datetime.isoformat('T'))
    # print('PSNR (original)', hmi_psnro, iti_psnro)
    # print('MSE', hmi_mse, iti_mse)
    # print('PSNR', hmi_psnr, iti_psnr)
    # print('SSIM', hmi_ssim, iti_ssim)
    # print('RMS contrast', hinode_rmsc, hmi_rmsc, iti_rmsc)
    # print('STAT MEAN', np.nanmean(hmi_data) / np.nanmean(hinode_data), np.nanmean(iti_data) / np.nanmean(hinode_data), )
    # print('STAT STD', np.nanstd(hmi_data) / np.nanstd(hinode_data), np.nanstd(iti_data) / np.nanstd(hinode_data), )
    # print('SUCCESS', transformation_hmi['success'], transformation_iti['success'])
    #
    iti_data = iti_data / 5e4
    hinode_registered_iti = hinode_registered_iti / 5e4
    hmi_data = hmi_data / 5e4
    hinode_registered_hmi = hinode_registered_hmi / 5e4

    evaluation['iti']['ssim'] += [ssim(iti_data, hinode_registered_iti)]
    evaluation['iti']['psnr'] += [psnr(iti_data, hinode_registered_iti)]
    evaluation['iti']['rmsc'] += [rms_contrast_diff(iti_data, hinode_registered_iti)]
    evaluation['iti']['mae'] += [mae(iti_data, hinode_registered_iti)]
    evaluation['iti']['cc'] += [image_correlation(iti_data, hinode_registered_iti)]
    evaluation['hmi']['ssim'] += [ssim(hmi_data, hinode_registered_hmi)]
    evaluation['hmi']['psnr'] += [psnr(hmi_data, hinode_registered_hmi)]
    evaluation['hmi']['rmsc'] += [rms_contrast_diff(hmi_data, hinode_registered_hmi)]
    evaluation['hmi']['mae'] += [mae(hmi_data, hinode_registered_hmi)]
    evaluation['hmi']['cc'] += [image_correlation(hmi_data, hinode_registered_hmi)]

    [print("ITI", k, np.mean(v)) for k, v in evaluation['iti'].items()]
    [print("HMI", k, np.mean(v)) for k, v in evaluation['hmi'].items()]

fid_iti = calculate_fid_given_paths((hinode_evaluation_path, iti_evaluation_path), 4,
                                    'cuda', 2048)
fid_hmi = calculate_fid_given_paths((hinode_evaluation_path, hmi_evaluation_path), 4,
                                    'cuda', 2048)

with open(os.path.join(evaluation_path, 'evaluation.txt'), 'w') as f:
    print('(ssim, psnr, mae, rmsc, FID, CC)', file=f)
    print('ITI', file=f)
    print('(%f, %f, %f, %f, %f, %f)' % (
        np.mean(evaluation['iti']['ssim']), np.mean(evaluation['iti']['psnr']), np.mean(evaluation['iti']['mae']),
        np.mean(evaluation['iti']['rmsc']), fid_iti, np.mean(evaluation['iti']['cc'])), file=f)
    print('HMI', file=f)
    print('(%f, %f, %f, %f, %f, %f)' % (
        np.mean(evaluation['hmi']['ssim']), np.mean(evaluation['hmi']['psnr']), np.mean(evaluation['hmi']['mae']),
        np.mean(evaluation['hmi']['rmsc']), fid_hmi, np.mean(evaluation['hmi']['cc'])), file=f)
