import glob
import os
import traceback
from datetime import datetime

from astropy.io.fits import getdata, getheader
from dateutil.parser import parse
from skimage import transform
from skimage.feature import ORB, match_descriptors
from skimage.feature import register_translation
from skimage.metrics import structural_similarity
from sklearn.utils import shuffle
from sunpy.map import Map

import matplotlib.pylab as pylab
from iti.download.hmi_continuum_download import HMIContinuumFetcher

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.util import view_as_blocks
from tqdm import tqdm

from iti.data.dataset import HMIContinuumDataset, HinodeDataset
from iti.data.editor import PaddingEditor, ScaleEditor, DataToMapEditor, ReductionEditor
from iti.train.trainer import Trainer

from astropy import units as u

hmi_shape = 4096
patch_shape = 1024
n_patches = hmi_shape // patch_shape
base_path = '/gss/r.jarolim/iti/hmi_hinode_v12'
evaluation_path = os.path.join(base_path, "comparison")
data_path = os.path.join(evaluation_path, "data")
os.makedirs(data_path, exist_ok=True)

hinode_files = shuffle(glob.glob('/gss/r.jarolim/data/hinode/level1/*.fits'))
special_features = []
quite_sun = []
for f in tqdm(hinode_files):
    data = np.ravel(getdata(f)) / getheader(f)['EXPTIME']
    if np.sum(data < 25000) > 2000:
        special_features.append(f)
    if len(special_features) >= 10:
        break

hinode_sample = ['/gss/r.jarolim/data/hinode/level1/FG20141022_141026.5.fits']#special_features  # + ['/gss/r.jarolim/data/hinode/level1/FG20141022_141026.5.fits']
hinode_dates = [parse(f[-22:-7].replace('_', 'T')) for f in hinode_sample]

fetcher = HMIContinuumFetcher(ds_path=data_path)
fetcher.fetchDates(hinode_dates)

hmi_files = glob.glob(os.path.join(data_path, "6173/**.fits"), recursive=True)
data_set = [(d, f) for f in hmi_files for d in hinode_dates if d.isoformat('T') in os.path.basename(f)]
hmi_data = [d[1] for d in data_set]
hinode_data = [f for d, _ in data_set for h_d, f in zip(hinode_dates, hinode_sample) if d == h_d]

hinode_dataset = HinodeDataset(hinode_data)
hinode_dataset.addEditor(DataToMapEditor())

hmi_dataset = HMIContinuumDataset(hmi_data)
hmi_dataset.addEditor(DataToMapEditor())
padding_editor = PaddingEditor((hmi_shape, hmi_shape))
recudction_editor = ReductionEditor()

trainer = Trainer(1, 1, upsampling=2, norm='in_rs_aff', lambda_diversity=0)
trainer.cuda()
iteration = trainer.resume(base_path, epoch=160000)
print('Loaded Iteration %d' % iteration)


def convertHMIMap(hmi_img, hmi_f):
    hmi_map = Map(hmi_f)
    hmi_map = ScaleEditor(0.6).call(hmi_map)
    hmi_map = Map(hmi_img[0], hmi_map.meta)
    return hmi_map


def convertHinodeMap(hinode_img, hinode_f):
    hinode_map = Map(hinode_f)
    hinode_map = ScaleEditor(0.15).call(hinode_map)
    hinode_map = Map(hinode_img[0], hinode_map.meta)
    return hinode_map


def prepImages(image, image_ref):
    min_x = min(image.shape[0], image_ref.shape[0])
    min_y = min(image.shape[1], image_ref.shape[1])
    image, image_ref = image[:min_x, :min_y], image_ref[:min_x, :min_y]
    return image, image_ref


def normalize_image(image):
    # image = (image - np.mean(image)) / np.std(image)
    image = (image + 1) / 2  # (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def contrast_normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def getShift(image, image_ref):
    image, image_ref = prepImages(image, image_ref)
    shift, _, _ = register_translation(image, image_ref)
    return np.flip(shift)


def getRotation(image, image_ref):
    image, image_ref = prepImages(image, image_ref)
    image, image_ref = image.astype(np.double), image_ref.astype(np.double)
    descriptor_extractor = ORB(n_keypoints=1000)
    #
    descriptor_extractor.detect_and_extract(image)
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors
    #
    descriptor_extractor.detect_and_extract(image_ref)
    keypoints_ref = descriptor_extractor.keypoints
    descriptors_ref = descriptor_extractor.descriptors
    #
    matches = match_descriptors(descriptors, descriptors_ref, cross_check=True)
    src, dst = keypoints[matches[:, 0]], keypoints_ref[matches[:, 1]]
    tform = transform.estimate_transform('euclidean', src, dst)
    return tform.rotation * 180 / np.pi


def enhanceHMI(hmi_img):
    hmi_patches = view_as_blocks(hmi_img, (patch_shape, patch_shape))
    hmi_patches = np.reshape(hmi_patches, (-1, patch_shape, patch_shape))
    enhanced_patches = []
    with torch.no_grad():
        for hmi_patch in tqdm(hmi_patches):
            enhanced_patch = trainer.forwardAB(
                torch.tensor(hmi_patch).float().cuda().view((1, 1, patch_shape, patch_shape)))
            enhanced_patches.append(enhanced_patch[0, 0].detach().cpu().numpy())
    #
    enhanced_patches = np.array(enhanced_patches).reshape((n_patches, n_patches, hmi_shape, hmi_shape))
    enhanced_img = np.array(enhanced_patches).transpose(0, 2, 1, 3).reshape(-1, enhanced_patches.shape[1] *
                                                                            enhanced_patches.shape[3])
    #
    return enhanced_img


params = {'axes.labelsize': 'large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

for hmi_map, hinode_map in zip(hmi_dataset, hinode_dataset):
    try:
        hmi_map = hmi_map.rotate(recenter=True, missing=-1)
        #hinode_map = hinode_map.rotate(recenter=False, missing=-1)
        #
        hmi_img = padding_editor.call(hmi_map.data)
        iti_img = enhanceHMI(hmi_img)
        target_shape = tuple(d * 4 for d in hmi_map.data.shape)
        iti_img = recudction_editor.call(iti_img, patch_shape=target_shape)
        upsampled_hmi_map = hmi_map.resample(target_shape * u.pix)
        iti_map = Map(iti_img, upsampled_hmi_map.meta)
        #
        upsampled_hmi_map = Map(normalize_image(upsampled_hmi_map.data), upsampled_hmi_map.meta)
        hinode_map = Map(normalize_image(hinode_map.data), hinode_map.meta)
        iti_map = Map(normalize_image(iti_map.data), iti_map.meta)
        #
        submap = upsampled_hmi_map.submap(hinode_map.bottom_left_coord, hinode_map.top_right_coord)
        shift = getShift(submap.data, hinode_map.data)
        print('Shift:', shift)
        hinode_map.meta['crpix1'] -= shift[0]
        hinode_map.meta['crpix2'] -= shift[1]
        #
        submap = upsampled_hmi_map.submap(hinode_map.bottom_left_coord, hinode_map.top_right_coord)
        image, image_ref = prepImages(submap.data, hinode_map.data)
        #image, image_ref = contrast_normalize(image), contrast_normalize(image_ref)
        difference_hmi = np.abs(image - image_ref)
        hmi_ssim = structural_similarity(image, image_ref, data_range=1)
        difference_hmi_map = Map(difference_hmi, hinode_map.meta)
        #
        submap = iti_map.submap(hinode_map.bottom_left_coord, hinode_map.top_right_coord)
        image, image_ref = prepImages(submap.data, hinode_map.data)
        #image, image_ref = contrast_normalize(image), contrast_normalize(image_ref)
        difference_iti = np.abs(image - image_ref)
        iti_ssim = structural_similarity(image, image_ref, data_range=1)
        difference_iti_map = Map(difference_iti, hinode_map.meta)
        #
        vmin = hinode_map.min()
        vmax = hinode_map.max()
        date = hinode_map.date.datetime
        #
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.5), sharex=True, sharey=True)
        hmi_map.plot(axes=axs[0], cmap='gray', title=date.isoformat(' ', timespec='minutes'))
        axs[0].set_ylabel('Helioprojective Latitude [arcsec]')
        axs[0].set_xlabel('Helioprojective Longitude [arcsec]')
        iti_map.plot(axes=axs[1], cmap='gray' , title='')
        axs[1].set_ylabel('')
        axs[1].set_xlabel('Helioprojective Longitude [arcsec]')
        hinode_map.plot(axes=axs[2], cmap='gray', title='')
        axs[2].set_ylabel('')
        axs[2].set_xlabel('Helioprojective Longitude [arcsec]')
        # difference_hmi_map.plot(axes=axs[3], cmap='viridis',
        #                         title='Diff HMI; SSIM: %.03f; MSE: %.03f' % (hmi_ssim, np.mean(difference_hmi)), vmin=0,
        #                         vmax=1)
        # difference_iti_map.plot(axes=axs[4], cmap='viridis',
        #                         title='Diff ITI; SSIM: %.03f; MSE: %.03f' % (iti_ssim, np.mean(difference_iti)),
        #                         vmin=0, vmax=1)
        plt.tight_layout()
        plt.savefig(os.path.join(evaluation_path, 'hinode_comparison_%s.jpg' % date.isoformat('T')), dpi=300)
        plt.close()
        print('Diff HMI; SSIM: %.03f; MSE: %.03f' % (hmi_ssim, np.mean(difference_hmi)))
        print('Diff ITI; SSIM: %.03f; MSE: %.03f' % (iti_ssim, np.mean(difference_iti)))
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        continue
