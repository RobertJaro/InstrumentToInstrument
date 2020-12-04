import glob
import os
from datetime import timedelta
from random import sample

from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from matplotlib.colors import Normalize
from sunpy.map import Map
from sunpy.physics.differential_rotation import solar_rotate_coordinate

from iti.download.hmi_continuum_download import HMIContinuumFetcher

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from skimage.util import view_as_blocks
from torch.utils.data import DataLoader
from tqdm import tqdm

from iti.data.dataset import HMIContinuumDataset, HinodeDataset
from iti.data.editor import PaddingEditor, LoadMapEditor, ScaleEditor, ReductionEditor
from iti.train.trainer import Trainer

from astropy import units as u

hmi_shape = 4096
patch_shape = 1024
n_patches = hmi_shape // patch_shape
base_path = '/gss/r.jarolim/iti/hmi_hinode_v12'
evaluation_path = os.path.join(base_path, "video")
data_path = os.path.join(evaluation_path, "data")
zoom_in_path = os.path.join(evaluation_path, "zoom_in")
series_path = os.path.join(evaluation_path, "series")
os.makedirs(data_path, exist_ok=True)
os.makedirs(zoom_in_path, exist_ok=True)
os.makedirs(series_path, exist_ok=True)
shift = (-214, -33)

hinode_sample = '/gss/r.jarolim/data/hinode/level1/FG20141022_141026.5.fits'
hinode_date = parse(hinode_sample[-22:-7].replace('_', 'T'))
hinode_dates = [hinode_date - timedelta(seconds=45) * i for i in range(2 * 80)]


fetcher = HMIContinuumFetcher(ds_path=data_path, num_worker_threads=2, ignore_quality=True, series='hmi.Ic_45s')
fetcher.fetchDates(hinode_dates)

hmi_dataset = HMIContinuumDataset(data_path)
padding_editor = PaddingEditor((hmi_shape, hmi_shape))
reduction_editor = ReductionEditor()
loader = DataLoader(hmi_dataset, batch_size=1, shuffle=False)

trainer = Trainer(1, 1, upsampling=2, norm='in_rs_aff', lambda_diversity=0)
trainer.cuda()
iteration = trainer.resume(base_path, epoch=160000)
print('Loaded Iteration %d' % iteration)


with torch.no_grad():
    for f, hmi_img in tqdm(zip(hmi_dataset.data, loader), total=len(hmi_dataset)):
        reference_map, _ = LoadMapEditor().call(f)
        iti_map_path = os.path.join(evaluation_path, 'iti_%s.fits' % reference_map.date.to_datetime().isoformat('T'))
        if os.path.exists(iti_map_path):
            continue
        reference_map = ScaleEditor(0.6).call(reference_map)
        #
        hmi_img = hmi_img[0, 0].float().detach().numpy()
        init_shape = hmi_img.shape
        hmi_img = padding_editor.call(hmi_img)
        hmi_patches = view_as_blocks(hmi_img, (patch_shape, patch_shape))
        hmi_patches = np.reshape(hmi_patches, (-1, patch_shape, patch_shape))
        iti_patches = []
        for hmi_patch in hmi_patches:
            hinode_patch = trainer.forwardAB(torch.tensor(hmi_patch).cuda().view((1, 1, patch_shape, patch_shape)))
            iti_patches.append(hinode_patch[0, 0].detach().cpu().numpy())
        #
        #
        iti_patches = np.array(iti_patches).reshape((n_patches, n_patches, hmi_shape, hmi_shape))
        iti_img = np.array(iti_patches).transpose(0, 2, 1, 3).reshape(-1, iti_patches.shape[1] * iti_patches.shape[3])
        iti_img = reduction_editor.call(iti_img, patch_shape=[s * 4 for s in init_shape])
        #
        iti_map = reference_map.resample(iti_img.shape * u.pix)
        iti_map = Map(iti_img, iti_map.meta)
        iti_map.save(iti_map_path)

hinode_map = Map(hinode_sample)
hinode_map.meta['crpix1'] -= shift[0]
hinode_map.meta['crpix2'] -= shift[1]
bl_coord = hinode_map.bottom_left_coord

iti_maps = Map(sorted(glob.glob(os.path.join(evaluation_path, 'iti_*.fits'))), sequence=True)

lin_bl_x = np.linspace(iti_maps[0].top_right_coord.Tx.value, bl_coord.Tx.value, len(iti_maps) * 2 // 3).tolist()
lin_bl_y = np.linspace(iti_maps[0].top_right_coord.Ty.value, bl_coord.Ty.value, len(iti_maps) * 2 // 3).tolist()
lin_scale = np.linspace(iti_maps[0].data.shape[0], hinode_map.data.shape[0], len(iti_maps) * 2 // 3).tolist()

lin_bl_x += [lin_bl_x[-1]] * round(len(iti_maps) / 3)
lin_bl_y += [lin_bl_y[-1]] * round(len(iti_maps) / 3)
lin_scale += [lin_scale[-1]] * round(len(iti_maps) / 3)

for s_map, bl_x, bl_y, scale in zip(iti_maps, lin_bl_x, lin_bl_y, lin_scale):
    bl_coord = SkyCoord(bl_x * u.arcsec, bl_y * u.arcsec, frame=s_map.coordinate_frame)
    rotated = solar_rotate_coordinate(bl_coord, time=s_map.date - hinode_map.date)
    print(rotated.Tx, bl_coord.Tx)
    bl_coord = bl_coord if np.isnan(rotated.Tx.value) else rotated
    scale = s_map.scale[0] * (scale * u.pix)
    sub_map = s_map.submap(bottom_left=bl_coord, width=scale, height=scale)
    plt.figure(figsize=(8,8))
    plt.subplot(111, projection=sub_map)
    plt.imshow(sub_map.data, cmap=sub_map.plot_settings['cmap'])
    plt.ylabel('Helioprojective Latitude [arcsec]', fontsize=16)
    plt.xlabel('Helioprojective Longitude [arcsec]', fontsize=16)
    #plt.tight_layout(5)
    plt.savefig(os.path.join(zoom_in_path, '%s.jpg' % s_map.date.to_datetime().isoformat('T')), dpi=150)
    plt.close()

hmi_maps = Map(sorted(glob.glob(os.path.join(evaluation_path, 'data', '6173', '*.fits'))), sequence=True)

bl_x, bl_y, scale = lin_bl_x[-1], lin_bl_y[-1], lin_scale[-1]
for iti_map, hmi_map in zip(iti_maps, hmi_maps):
    bl_coord = SkyCoord(bl_x * u.arcsec, bl_y * u.arcsec, frame=iti_map.coordinate_frame)
    bl_coord = solar_rotate_coordinate(bl_coord, time=iti_map.date - hinode_map.date)
    s = iti_map.scale[0] * (scale * u.pix)
    iti_sub_map = iti_map.submap(bottom_left=bl_coord, width=s, height=s)
    hmi_sub_map = hmi_map.submap(bottom_left=bl_coord, width=s, height=s)
    plt.figure(figsize=(13, 6))
    plt.subplot(121, projection=hmi_sub_map)
    plt.imshow(hmi_sub_map.data, cmap=hmi_sub_map.plot_settings['cmap'])
    plt.ylabel('Helioprojective Latitude [arcsec]', fontsize=20)
    plt.xlabel('Helioprojective Longitude [arcsec]', fontsize=20)
    plt.subplot(122, projection=iti_sub_map)
    plt.imshow(iti_sub_map.data, cmap=iti_sub_map.plot_settings['cmap'])
    plt.ylabel('')
    plt.xlabel('Helioprojective Longitude [arcsec]', fontsize=20)
    plt.suptitle('%s' % iti_map.date.to_datetime().isoformat(' '), fontsize=24)
    #plt.tight_layout(pad=2)
    plt.savefig(os.path.join(series_path, '%s.jpg' % iti_map.date.to_datetime().isoformat('T')), dpi=300)
    plt.close()
