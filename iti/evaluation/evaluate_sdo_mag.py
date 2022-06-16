import os

import matplotlib.cm
import matplotlib.pyplot as plt
import torch
from imageio import imsave
from matplotlib.colors import Normalize
from skimage import draw
from skimage.metrics import structural_similarity
from sunpy.visualization.colormaps import cm
from torch.utils.data import DataLoader
from tqdm import tqdm

from iti.data.dataset import get_intersecting_files, SDODataset, StorageDataset
from iti.evaluation.compute_fid import calculate_fid_given_paths
from iti.train.model import DiscriminatorMode
from iti.trainer import Trainer



from iti.translate import STEREOToSDO

import numpy as np

# init
base_path = "/gpfs/gpfs0/robert.jarolim/iti/sdo_euv_to_mag_v1"
base_path_abs = "/gpfs/gpfs0/robert.jarolim/iti/sdo_euv_to_abs_mag_v1"
prediction_path = os.path.join(base_path, 'evaluation_check')
os.makedirs(prediction_path, exist_ok=True)

pix_path = '/beegfs/home/robert.jarolim/projects/Pix2Pix/evaluation/sdo_euv_to_mag_v2'

cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    matplotlib.cm.get_cmap('gray')
]

sdo_path = "/gpfs/gpfs0/robert.jarolim/data/iti/sdo"
wls = ['171', '193', '211', '304', '6173']
sdo_converted_path = '/gpfs/gpfs0/robert.jarolim/data/converted/sdo_512'
sdo_files = get_intersecting_files(sdo_path, wls, ext='.fits', months=[11, 12])
sdo_files = np.array(sdo_files)[:, ::100].tolist()
sdo_ds = SDODataset(sdo_files, resolution=512)
storage_ds = StorageDataset(sdo_ds, sdo_converted_path, ext_editors=[])
loader = DataLoader(storage_ds, batch_size=1, num_workers=4)

trainer = Trainer(4, 5, upsampling=0, discriminator_mode=DiscriminatorMode.CHANNELS, lambda_diversity=0,
                  norm='in_rs_aff')
trainer.cuda()
trainer.resume(base_path)
trainer.eval()

trainer_abs = Trainer(4, 5, upsampling=0, discriminator_mode=DiscriminatorMode.CHANNELS, lambda_diversity=0,
                      norm='in_rs_aff')
trainer_abs.cuda()
trainer_abs.resume(base_path_abs, 160000)
trainer_abs.eval()

diff_iti = []
diff_pix = []
abs_diff_iti = []
abs_diff_iti_abs = []
abs_diff_pix = []
abs_iti = []
abs_iti_abs = []
abs_pix = []
ssim_iti = []
ssim_pix = []
abs_ssim_iti_abs = []
abs_ssim_iti = []
abs_ssim_pix = []

img_iti_path = os.path.join(prediction_path, 'iti')
img_iti_abs_path = os.path.join(prediction_path, 'iti_abs')
img_sdo_path = os.path.join(prediction_path, 'sdo')
img_pix_path = os.path.join(prediction_path, 'pix')
os.makedirs(img_iti_path, exist_ok=True), os.makedirs(img_iti_abs_path, exist_ok=True)
os.makedirs(img_sdo_path, exist_ok=True), os.makedirs(img_pix_path, exist_ok=True)

for file, sdo_cube in tqdm(zip(sdo_files[0], loader), total=len(loader)):
    bn = os.path.basename(file).split('.')[0]
    pix_npy_path = os.path.join(pix_path, '%s.npy' % bn)
    if not os.path.exists(pix_npy_path):
        continue

    with torch.no_grad():
        sdo_cube = sdo_cube.cuda()
        iti_cube = trainer.forwardAB(sdo_cube[:, :-1])
        iti_abs_cube = trainer_abs.forwardAB(sdo_cube[:, :-1])
    sdo_cube = sdo_cube.detach().cpu().numpy()[0]
    iti_cube = iti_cube.detach().cpu().numpy()[0]
    iti_abs_cube = iti_abs_cube.detach().cpu().numpy()[0]
    #
    pix = np.load(pix_npy_path)
    # adjust abs magnetogram
    iti_abs_cube[-1] = (iti_abs_cube[-1] + 1) / 2
    # remove off limb
    yy, xx = np.mgrid[:512, :512]
    r = np.sqrt((xx - 256) ** 2 + (yy - 256) ** 2) / (256 / 1.1)
    iti_cube[-1, r >= 1] = -1 # np.nan
    iti_abs_cube[-1, r >= 1] = -1 # np.nan
    sdo_cube[-1, r >= 1] = -1# np.nan
    pix[r >= 1] = -1 # np.nan
    pix_diff = np.abs(np.abs(pix) - np.abs(sdo_cube[-1]))
    iti_diff = np.abs(np.abs(iti_cube[-1]) - np.abs(sdo_cube[-1]))
    # vmax = max(pix_diff.max(), iti_diff.max(), iti_abs_diff.max())
    # plt.imsave(os.path.join(img_iti_path, '%s.jpg' % bn), iti_cube[-1], cmap='gray', vmin=-1, vmax=1)
    # plt.imsave(os.path.join(img_iti_path, '%s_diff.jpg' % bn), iti_diff, cmap='inferno', vmin=0, vmax=vmax)
    # plt.imsave(os.path.join(img_iti_abs_path, '%s.jpg' % bn), (iti_abs_cube[-1] + 1) / 2, cmap='gray', vmin=-1, vmax=1)
    # plt.imsave(os.path.join(img_iti_abs_path, '%s_diff.jpg' % bn), iti_abs_diff, cmap='inferno', vmin=0, vmax=vmax)
    # plt.imsave(os.path.join(img_sdo_path, '%s.jpg' % bn), sdo_cube[-1], cmap='gray', vmin=-1, vmax=1)
    # plt.imsave(os.path.join(img_pix_path, '%s.jpg' % bn), pix, cmap='gray', vmin=-1, vmax=1)
    # plt.imsave(os.path.join(img_pix_path, '%s_diff.jpg' % bn), pix_diff, cmap='inferno', vmin=0, vmax=vmax)
    #
    diff_iti += [np.nanmean(np.abs(iti_cube[-1] - sdo_cube[-1]))]
    diff_pix += [np.nanmean(np.abs(pix - sdo_cube[-1]))]
    abs_diff_iti += [np.nanmean(np.abs(np.abs(iti_cube[-1]) - np.abs(sdo_cube[-1])))]
    abs_diff_iti_abs += [np.nanmean(np.abs(np.abs(iti_abs_cube[-1]) - np.abs(sdo_cube[-1])))]
    abs_diff_pix += [np.nanmean(np.abs(np.abs(pix) - np.abs(sdo_cube[-1])))]
    iti_cube[-1, r >= 1] = 0
    iti_abs_cube[-1, r >= 1] = 0
    sdo_cube[-1, r >= 1] = 0
    pix[r >= 1] = 0
    total_sdo_mag = np.abs(sdo_cube[-1]).sum()
    abs_iti += [np.abs(np.abs(iti_cube[-1]).sum() - total_sdo_mag)]
    abs_iti_abs += [np.abs((iti_abs_cube[-1]).sum() - total_sdo_mag)]
    abs_pix += [np.abs(np.abs(pix).sum() - total_sdo_mag)]
    ssim_iti += [structural_similarity(iti_cube[-1], sdo_cube[-1], data_range=2)]
    ssim_pix += [structural_similarity(pix, sdo_cube[-1], data_range=2)]
    abs_ssim_iti_abs += [structural_similarity(iti_abs_cube[-1], np.abs(sdo_cube[-1]), data_range=1)]
    abs_ssim_iti += [structural_similarity(np.abs(iti_cube[-1]), np.abs(sdo_cube[-1]), data_range=1)]
    abs_ssim_pix += [structural_similarity(np.abs(pix), np.abs(sdo_cube[-1]), data_range=1)]

fid_iti = 0#calculate_fid_given_paths((img_sdo_path, img_iti_path), 4, 'cuda', 2048)
fid_pix = 0#calculate_fid_given_paths((img_sdo_path, img_pix_path), 4, 'cuda', 2048)

with open(os.path.join(prediction_path, 'evaluation.txt'), 'w') as f:

    print('Difference ITI', np.mean(diff_iti) * 3000, file=f)
    print('Difference PIX', np.mean(diff_pix) * 3000, file=f)

    print('Abs Difference ITI', np.mean(abs_diff_iti) * 3000, file=f)
    print('Abs Difference ITI abs', np.mean(abs_diff_iti_abs) * 3000, file=f)
    print('Abs Difference PIX', np.mean(abs_diff_pix) * 3000, file=f)

    print('Abs ITI', np.mean(abs_iti) * 3000, file=f)
    print('Abs ITI abs', np.mean(abs_iti_abs) * 3000, file=f)
    print('Abs PIX', np.mean(abs_pix) * 3000, file=f)

    print('FID ITI', fid_iti, file=f)
    print('FID PIX', fid_pix, file=f)

    print('SSIM ITI', np.mean(ssim_iti), file=f)
    print('SSIM PIX', np.mean(ssim_pix), file=f)

    print('Abs SSIM ITI', np.mean(abs_ssim_iti), file=f)
    print('Abs SSIM ITI abs', np.mean(abs_ssim_iti_abs), file=f)
    print('Abs SSIM PIX', np.mean(abs_ssim_pix), file=f)
