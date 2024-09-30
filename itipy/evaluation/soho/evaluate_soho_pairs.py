import glob
import os
from datetime import timedelta
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from dateutil.parser import parse
from imreg_dft import similarity, transform_img_dict
from matplotlib.colors import Normalize
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from itipy.data.baseline_calibration import aia_calibration, eit_calibration
from itipy.data.editor import sdo_norms, RemoveOffLimbEditor
from itipy.data.loader import HMIMapLoader, AIAMapLoader, EITMapLoader, MDIMapLoader
from itipy.evaluation.metrics import ssim, psnr, image_correlation
from itipy.evaluation.register import register
from itipy.translate import SOHOToSDO

# init
base_path = "/gpfs/gpfs0/robert.jarolim/iti/soho_sdo_v4"
soho_data_path = '/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep'
sdo_data_path = '/gpfs/gpfs0/robert.jarolim/data/iti/sdo_comparison_iti2022'

prediction_path = '/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v2'
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = SOHOToSDO(model_path=os.path.join(base_path, 'generator_AB.pt'))

# translate
basenames_soho = [[os.path.basename(f) for f in glob.glob('%s/%s/*.fits' % (soho_data_path, wl))] for
                  wl in ['171', '195', '284', '304', 'mag']]
basenames_soho = set(basenames_soho[0]).intersection(*basenames_soho[1:])
basenames_sdo = [[os.path.basename(f) for f in glob.glob('%s/%s/*.fits' % (sdo_data_path, wl))] for wl in
                 ['171', '193', '211', '304', '6173']]
basenames_sdo = set(basenames_sdo[0]).intersection(*basenames_sdo[1:])

dates_soho = sorted([parse(f.split('.')[0]) for f in basenames_soho])
dates_sdo = sorted([parse(f.split('.')[0]) for f in basenames_sdo])

closest_dates = [(date_soho, min(dates_sdo, key=lambda x: abs(x - date_soho))) for date_soho in dates_soho]
selected_dates = [(date_soho, date_sdo) for date_soho, date_sdo in closest_dates if
                  np.abs(date_soho - date_sdo) < timedelta(hours=1)]  # file name filter (below filter < 1 min)

selected_dates = selected_dates[::3]

basenames_soho = ['%s.fits' % date_soho.isoformat('T') for date_soho, date_sdo in selected_dates]
basenames_sdo = ['%s.fits' % date_sdo.isoformat('T') for date_soho, date_sdo in selected_dates]

iti_maps = translator.translate(soho_data_path, basenames=basenames_soho)
soho_maps = ([Map('%s/%s/%s' % (soho_data_path, dir, basename))
              for dir in ['171', '195', '284', '304', 'mag']]
             for basename in basenames_soho)
sdo_maps = ([Map('%s/%s/%s' % (sdo_data_path, dir, basename))
             for dir in ['171', '193', '211', '304', '6173']] for basename in basenames_sdo)

aia_loader = AIAMapLoader()
hmi_loader = HMIMapLoader()
eit_loader = EITMapLoader()
mdi_loader = MDIMapLoader()

off_limb_editor = RemoveOffLimbEditor()




norms = [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304], sdo_norms['mag']]

evaluation = {'171': {'soho': {'ssim': [], 'psnr': [], 'cc': []}, 'iti': {'ssim': [], 'psnr': [], 'cc': []}},
              '193': {'soho': {'ssim': [], 'psnr': [], 'cc': []}, 'iti': {'ssim': [], 'psnr': [], 'cc': []}},
              '211': {'soho': {'ssim': [], 'psnr': [], 'cc': []}, 'iti': {'ssim': [], 'psnr': [], 'cc': []}},
              '304': {'soho': {'ssim': [], 'psnr': [], 'cc': []}, 'iti': {'ssim': [], 'psnr': [], 'cc': []}},
              'mag': {'soho': {'ssim': [], 'psnr': [], 'cc': []}, 'iti': {'ssim': [], 'psnr': [], 'cc': []}}
              }

for soho_cube, iti_cube, sdo_cube in tqdm(zip(soho_maps, iti_maps, sdo_maps), total=len(selected_dates)):
    date = soho_cube[-1].date.datetime
    if np.abs(soho_cube[-1].date.datetime - sdo_cube[-1].date.datetime) > timedelta(minutes=15):
        print('Invalid!', np.abs(soho_cube[-1].date.datetime - sdo_cube[-1].date.datetime))
        continue
    simplefilter('ignore')  # ignore int conversion warning
    for i in range(4):
        idx = list(aia_calibration.keys())[i]
        sdo_map = aia_loader(sdo_cube[i])
        soho_map = eit_loader(soho_cube[i])
        iti_map = iti_cube[i]
        #
        eit_mean, eit_std = list(eit_calibration.values())[i]['mean'], list(eit_calibration.values())[i]['std']
        aia_mean, aia_std = list(aia_calibration.values())[i]['mean'], list(aia_calibration.values())[i]['std']
        norm = norms[i]
        norm = Normalize(vmin=0, vmax=norm.vmax, clip=True)

        iti_data = norm(iti_map.data)
        sdo_data = norm(sdo_map.data)

        soho_map = soho_map.resample(sdo_data.shape * u.pix)
        soho_data = soho_map.data
        soho_data = (soho_data - eit_mean) * (aia_std / eit_std) + aia_mean
        soho_data = norm(soho_data)

        soho_data = register(soho_data, sdo_data, 0, strides=8)
        iti_data = register(iti_data, sdo_data, 0, strides=8)

        # remove padding
        soho_data = soho_data[32:-32, 32:-32]
        iti_data = iti_data[32:-32, 32:-32]
        sdo_data = sdo_data[32:-32, 32:-32]

        soho_ssim = ssim(soho_data, sdo_data)
        soho_psnr = psnr(soho_data, sdo_data)
        soho_cc = image_correlation(soho_data, sdo_data)

        iti_ssim = ssim(iti_data, sdo_data)
        iti_psnr = psnr(iti_data, sdo_data)
        iti_cc = image_correlation(iti_data, sdo_data)

        evaluation[idx]['soho']['ssim'].append(soho_ssim)
        evaluation[idx]['soho']['psnr'].append(soho_psnr)
        evaluation[idx]['soho']['cc'].append(soho_cc)
        evaluation[idx]['iti']['ssim'].append(iti_ssim)
        evaluation[idx]['iti']['psnr'].append(iti_psnr)
        evaluation[idx]['iti']['cc'].append(iti_cc)

        print(idx)
        print('SOHO: SSIM %.03f; PSNR %.03f; CC %.03f;' %
              (np.mean(evaluation[idx]['soho']['ssim']), np.mean(evaluation[idx]['soho']['psnr']),
               np.mean(evaluation[idx]['soho']['cc'])))
        print('ITI: SSIM %.03f; PSNR %.03f; CC %.03f;' %
              (np.mean(evaluation[idx]['iti']['ssim']), np.mean(evaluation[idx]['iti']['psnr']),
               np.mean(evaluation[idx]['iti']['cc'])))

        img_norm = Normalize(vmin=0, vmax=np.nanmax(sdo_data[768:-768, 768:-768]), clip=True)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        cmap = sdo_map.plot_settings['cmap']
        axs[0].imshow(soho_data[768:-768, 768:-768], cmap=cmap, norm=img_norm)
        axs[0].set_title('SOHO')
        axs[1].imshow(iti_data[768:-768, 768:-768], cmap=cmap, norm=img_norm)
        axs[1].set_title('ITI')
        axs[2].imshow(sdo_data[768:-768, 768:-768], cmap=cmap, norm=img_norm)
        axs[2].set_title('SDO')
        fig.tight_layout()
        fig.savefig(os.path.join(prediction_path, '%s_%s.jpg' % (date.isoformat(), list(eit_calibration.keys())[i])))
        plt.close(fig)

    # evaluate magnetograms
    idx = 'mag'
    sdo_map = hmi_loader(sdo_cube[-1])
    soho_map = mdi_loader(soho_cube[-1])
    iti_map = iti_cube[-1]

    # remove off disk
    sdo_map = off_limb_editor.call(sdo_map)
    soho_map = off_limb_editor.call(soho_map)
    iti_map = off_limb_editor.call(iti_map)

    sdo_data = sdo_map.data / 1500
    iti_data = iti_map.data / 1500

    soho_map = soho_map.resample(sdo_data.shape * u.pix)
    soho_data = soho_map.data / 1500

    # clip value range
    sdo_data = np.clip(sdo_data, -1, 1)
    iti_data = np.clip(iti_data, -1, 1)
    soho_data = np.clip(soho_data, -1, 1)

    soho_data = register(soho_data, sdo_data, 0, strides=8)
    iti_data = register(iti_data, sdo_data, 0, strides=8)

    soho_ssim = ssim(soho_data, sdo_data, data_range=2)
    soho_psnr = psnr(soho_data, sdo_data, data_range=2)
    soho_cc = image_correlation(soho_data, sdo_data)

    iti_ssim = ssim(iti_data, sdo_data, data_range=2)
    iti_psnr = psnr(iti_data, sdo_data, data_range=2)
    iti_cc = image_correlation(iti_data, sdo_data)

    evaluation[idx]['soho']['ssim'].append(soho_ssim)
    evaluation[idx]['soho']['psnr'].append(soho_psnr)
    evaluation[idx]['soho']['cc'].append(soho_cc)
    evaluation[idx]['iti']['ssim'].append(iti_ssim)
    evaluation[idx]['iti']['psnr'].append(iti_psnr)
    evaluation[idx]['iti']['cc'].append(iti_cc)

    print(idx)
    print('SOHO: SSIM %.03f; PSNR %.03f; CC %.03f;' %
          (np.mean(evaluation[idx]['soho']['ssim']), np.mean(evaluation[idx]['soho']['psnr']),
           np.mean(evaluation[idx]['soho']['cc'])))
    print('ITI: SSIM %.03f; PSNR %.03f; CC %.03f;' %
          (np.mean(evaluation[idx]['iti']['ssim']), np.mean(evaluation[idx]['iti']['psnr']),
           np.mean(evaluation[idx]['iti']['cc'])))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    cmap = sdo_map.plot_settings['cmap']
    axs[0].imshow(soho_data, cmap=cm.hmimag, vmin=-1, vmax=1)
    axs[0].set_title('SOHO')
    axs[1].imshow(iti_data, cmap=cm.hmimag, vmin=-1, vmax=1)
    axs[1].set_title('ITI')
    axs[2].imshow(sdo_data, cmap=cm.hmimag, vmin=-1, vmax=1)
    axs[2].set_title('SDO')
    fig.tight_layout()
    fig.savefig(os.path.join(prediction_path, '%s_%s.jpg' % (date.isoformat(), 'mag')))
    plt.close(fig)

with open(os.path.join(prediction_path, 'evaluation.txt'), 'w') as f:
    print('SOHO', file=f)
    for key, value in evaluation.items():
        print(key, file=f)
        print('SSIM %.03f; PSNR %.03f; CC %.03f;' % (
            np.mean(value['soho']['ssim']), np.mean(value['soho']['psnr']), np.mean(value['soho']['cc']))
              , file=f)
    print('ITI', file=f)
    for key, value in evaluation.items():
        print(key, file=f)
        print('SSIM %.03f; PSNR %.03f; CC %.03f;' % (
            np.mean(value['iti']['ssim']), np.mean(value['iti']['psnr']), np.mean(value['iti']['cc']))
              , file=f)
