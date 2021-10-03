import glob

import os

from scipy.fftpack import fft2, fftshift
from skimage.metrics import structural_similarity

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from sunpy.map import Map

from iti.translate import KSOLowToHigh, KSOFlatConverter

from matplotlib import pyplot as plt
import numpy as np

# init
base_path = '/gss/r.jarolim/iti/kso_quality_1024_v6'
prediction_path = os.path.join(base_path, 'compare')
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = KSOLowToHigh(resolution=1024, model_path=os.path.join(base_path, 'generator_AB.pt'))
converter = KSOFlatConverter(1024)


# load maps
map_files = list(glob.glob('/gss/r.jarolim/data/kso_comparison_iti2021/**/*.fts.gz', recursive=True))
lq_files = sorted([f for f in map_files if 'ref_' not in os.path.basename(f)])
hq_files = sorted([f for f in map_files if 'ref_' in os.path.basename(f)])
dates = [Map(f).date.datetime for f in lq_files]
ref_imgs, metas = converter.convert(hq_files)


def saveimg(img, path):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.set_axis_off()
    ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.tight_layout(0)
    fig.savefig(path, dpi=300)
    plt.close(fig)


# translate
result = []
for (_, kso_img, iti_img), ref_img, date in zip(translator.translate(lq_files), ref_imgs, dates):
    saveimg(np.array(kso_img)[0], os.path.join(prediction_path, '%s_kso.jpg' % date.isoformat()))
    saveimg(iti_img[0], os.path.join(prediction_path, '%s_iti.jpg' % date.isoformat()))
    saveimg(ref_img, os.path.join(prediction_path, '%s_ref.jpg' % date.isoformat()))
    #
    ssim_iti, ssim_kso = structural_similarity(iti_img[0], ref_img), structural_similarity(kso_img[0], ref_img)
    mse_iti, mse_kso = np.mean((iti_img[0] - ref_img) ** 2), np.mean((kso_img[0] - ref_img) ** 2)
    result.append((ssim_kso, mse_kso, ssim_iti, mse_iti))
    #
    kso_f = np.abs(fftshift(fft2((kso_img[0]))))
    iti_f = np.abs(fftshift(fft2((iti_img[0]))))
    ref_f = np.abs(fftshift(fft2((ref_img))))
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    [ax.set_axis_off() for ax in np.ravel(axs)]
    axs[0].imshow(np.log(kso_f), cmap='magma')
    axs[1].imshow(np.log(iti_f), cmap='magma')
    axs[2].imshow(np.log(ref_f), cmap='magma')
    fig.tight_layout(0)
    fig.savefig(os.path.join(prediction_path, '%s_fft.jpg' % date.isoformat()), dpi=100)
    plt.close(fig)


result = np.array(result)
print('SSIM: KSO %.03f; ITI %0.3f' % (np.mean(result[:, 0]), np.mean(result[:, 2])))
print('MSE: KSO %.03f; ITI %0.3f' % (np.mean(result[:, 1]), np.mean(result[:, 3])))

# v6
# SSIM: KSO 0.364; ITI 0.329
# MSE: KSO 0.020; ITI 0.023