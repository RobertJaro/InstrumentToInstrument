import glob
import os

from sunpy.coordinates import propagate_with_solar_surface

from itipy.evaluation.metrics import ssim, psnr, image_correlation
from itipy.evaluation.register import register

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from sunpy.map import Map

from itipy.translate import KSOLowToHigh
from itipy.data.loader import KSOFlatLoader

from matplotlib import pyplot as plt
import numpy as np

# init
prediction_path = '/beegfs/home/robert.jarolim/iti_evaluation/kso_v3'
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = KSOLowToHigh(resolution=1024, model_path='/gpfs/gpfs0/robert.jarolim/iti/kso_quality_v1/generator_AB.pt')
converter = KSOFlatLoader(1024)

valid_ref_files = [
                # 'ref_kanz_halph_fi_20121003_065519.fts.gz',
               'ref_kanz_halph_fi_20121003_131112.fts.gz',
               'ref_kanz_halph_fi_20130617_074400.fts.gz',
               'ref_kanz_halph_fi_20131125_125802.fts.gz',
               'ref_kanz_halph_fi_20160712_100052.fts.gz',
               'ref_kanz_halph_fi_20160712_132927.fts.gz',
               'ref_kanz_halph_fi_20160804_090315.fts.gz',
               'ref_kanz_halph_fi_20161025_074256.fts.gz',
               'ref_kanz_halph_fi_20170704_120538.fts.gz',
               'ref_kanz_halph_fi_20170922_140618.fts.gz',
               'ref_kanz_halph_fi_20170922_160738.fts.gz',
               'ref_kanz_halph_fi_20171226_094035.fts.gz',
               'ref_kanz_halph_fi_20180306_121738.fts.gz',
               'ref_kanz_halph_fi_20180607_102605.fts.gz',
               'ref_kanz_halph_fi_20200427_083416.fts.gz',
               'ref_kanz_halph_fi_20200427_083631.fts.gz',
               'ref_kanz_halph_fi_20201026_091034.fts.gz',
                   ]
valid_kso_files = [f.replace('ref_', '') for f in valid_ref_files]

# load maps
map_files = list(glob.glob('/beegfs/home/robert.jarolim/data/kso_comparison_iti2021/**/*.fts.gz', recursive=True))
map_files = [f for f in map_files if os.path.basename(f) in valid_ref_files or os.path.basename(f) in valid_kso_files]

lq_files = sorted([f for f in map_files if 'ref_' not in os.path.basename(f)])
hq_files = sorted([f for f in map_files if 'ref_' in os.path.basename(f)])
hq_maps = [Map(f) for f in hq_files]
dates = [Map(f).date.datetime for f in lq_files]

# translate
result = {'kso': {'ssim': [], 'psnr': [], 'cc': []},
          'iti': {'ssim': [], 'psnr': [], 'cc': []}
          }
for (iti_map, kso_img, iti_img), hq_map, date in zip(translator.translate(lq_files), hq_maps, dates):
    ref_map = converter(hq_map)
    kso_map = Map(kso_img[0], iti_map.meta)

    with propagate_with_solar_surface():
        kso_map, footprint = kso_map.reproject_to(ref_map.wcs, return_footprint=True)
        iti_map = iti_map.reproject_to(ref_map.wcs)
        kso_map.data[footprint == 0] = np.nan
        iti_map.data[footprint == 0] = np.nan
        ref_map.data[footprint == 0] = np.nan

    ref_img = ref_map.data
    kso_img = kso_map.data
    iti_img = iti_map.data

    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1, projection=ref_map)
    ref_map.plot(vmin=-1, vmax=1)
    ref_map.draw_grid()
    plt.title('Reference')
    plt.subplot(1, 3, 2, projection=iti_map)
    iti_map.plot(vmin=-1, vmax=1)
    iti_map.draw_grid()
    plt.title('ITI')
    plt.subplot(1, 3, 3, projection=iti_map)
    kso_map.plot(vmin=-1, vmax=1)
    kso_map.draw_grid()
    plt.title('KSO')
    plt.savefig(os.path.join(prediction_path, '%s_comparison.jpg' % date.isoformat()))
    plt.close(fig)

    kso_img = kso_img[256:-256, 256:-256]
    iti_img = iti_img[256:-256, 256:-256]
    ref_img = ref_img[256:-256, 256:-256]

    kso_img = register(kso_img, ref_img, constraints={'scale': (1, 0), 'angle': (0, 10)}, strides=1)
    iti_img = register(iti_img, ref_img, constraints={'scale': (1, 0), 'angle': (0, 10)}, strides=1)

    kso_img = kso_img[32:-32, 32:-32]
    iti_img = iti_img[32:-32, 32:-32]
    ref_img = ref_img[32:-32, 32:-32]

    plt.imsave(os.path.join(prediction_path, '%s_kso.jpg' % date.isoformat()), kso_img, cmap='gray', vmin=-1, vmax=1)
    plt.imsave(os.path.join(prediction_path, '%s_iti.jpg' % date.isoformat()), iti_img, cmap='gray', vmin=-1, vmax=1)
    plt.imsave(os.path.join(prediction_path, '%s_ref.jpg' % date.isoformat()), ref_img, cmap='gray', vmin=-1, vmax=1)
    #
    kso_img = (kso_img + 1) / 2
    iti_img = (iti_img + 1) / 2
    ref_img = (ref_img + 1) / 2

    iti_psnr = psnr(iti_img, ref_img)
    iti_cc = image_correlation(iti_img, ref_img)
    kso_psnr = psnr(kso_img, ref_img)
    kso_cc = image_correlation(kso_img, ref_img)

    iti_ssim = ssim(np.nan_to_num(iti_img, nan=0), np.nan_to_num(ref_img, nan=0))
    kso_ssim = ssim(np.nan_to_num(kso_img, nan=0), np.nan_to_num(ref_img, nan=0))

    result['kso']['ssim'].append(kso_ssim)
    result['kso']['psnr'].append(kso_psnr)
    result['kso']['cc'].append(kso_cc)

    result['iti']['ssim'].append(iti_ssim)
    result['iti']['psnr'].append(iti_psnr)
    result['iti']['cc'].append(iti_cc)

    print(date)
    print('TIME DIFFERENCE: %s' % np.abs(ref_map.date.datetime - iti_map.date.datetime))
    print('SSIM: KSO %.03f; ITI %.03f' % (kso_ssim, iti_ssim))
    print('PSNR: KSO %.03f; ITI %.03f' % (kso_psnr, iti_psnr))
    print('CC: KSO %.03f; ITI %.03f' % (kso_cc, iti_cc))

    print('RUNNING AVERAGE:')
    print('SSIM: KSO %.03f; ITI %.03f' % (np.mean(result['kso']['ssim']), np.mean(result['iti']['ssim'])))
    print('PSNR: KSO %.03f; ITI %.03f' % (np.mean(result['kso']['psnr']), np.mean(result['iti']['psnr'])))
    print('CC: KSO %.03f; ITI %.03f' % (np.mean(result['kso']['cc']), np.mean(result['iti']['cc'])))

    if kso_psnr > iti_psnr: # in most cases registration did not work
        print('CHECK THIS DATE: %s' % date)

with open(os.path.join(prediction_path, 'evaluation.txt'), 'w') as f:
    print('KSO', file=f)
    print('SSIM %.03f; PSNR %.03f; CC %.03f;' % (
        np.mean(result['kso']['ssim']), np.mean(result['kso']['psnr']), np.mean(result['kso']['cc']))
          , file=f)
    print('ITI', file=f)
    print('SSIM %.03f; PSNR %.03f; CC %.03f;' % (
        np.mean(result['iti']['ssim']), np.mean(result['iti']['psnr']), np.mean(result['iti']['cc']))
          , file=f)

# v6
# SSIM: KSO 0.364; ITI 0.329
# MSE: KSO 0.020; ITI 0.023
