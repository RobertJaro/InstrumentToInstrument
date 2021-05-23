import glob
import os

import matplotlib.pylab as pylab
from dateutil.parser import parse
from skimage import transform
from skimage.feature import ORB, match_descriptors
from skimage.feature import register_translation
from skimage.metrics import structural_similarity
from skimage.util import view_as_windows
from sunpy.map import Map

from iti.download.download_hmi_continuum import HMIContinuumDownloader
from iti.prediction.translate import HMIToHinode

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
from matplotlib import pyplot as plt

from iti.data.editor import ScaleEditor

from astropy import units as u

hmi_shape = 4096
patch_shape = 1024
n_patches = hmi_shape // patch_shape
base_path = '/gss/r.jarolim/iti/hmi_hinode_v12'
evaluation_path = os.path.join(base_path, "comparison")
data_path = os.path.join(evaluation_path, "data")
os.makedirs(data_path, exist_ok=True)

# hinode_files = shuffle(glob.glob('/gss/r.jarolim/data/hinode/level1/*.fits'))
# special_features = []
# quite_sun = []
# for f in tqdm(hinode_files):
#     data = np.ravel(getdata(f)) / getheader(f)['EXPTIME']
#     if np.sum(data < 25000) > 2000:
#         special_features.append(f)
#     if len(special_features) >= 10:
#         break

hinode_paths = [
    '/gss/r.jarolim/data/hinode/level1/FG20141022_141026.5.fits']  # special_features  # + ['/gss/r.jarolim/data/hinode/level1/FG20141022_141026.5.fits']
hinode_dates = [parse(f[-22:-7].replace('_', 'T')) for f in hinode_paths]

fetcher = HMIContinuumDownloader(ds_path=data_path)
fetcher.fetchDates(hinode_dates)
hmi_paths = sorted(glob.glob(os.path.join(data_path, '6173/*.fits')))

iti_model = HMIToHinode(model_path=os.path.join(base_path, 'generator_AB.pt'), patch_factor=3)

iti_maps = iti_model.translate(hmi_paths)
hinode_maps = [Map(path) for path in hinode_paths]
hinode_maps = [Map(m.data / m.exposure_time.to(u.s).value, m.meta) for m in hinode_maps] # normalize exposure time
hmi_maps = [Map(path) for path in hmi_paths]


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
    diff = np.abs(image - image_ref)
    max_pos = np.where(diff == diff.max())
    print(max_pos)
    y = min(int(max_pos[0]), diff.shape[0] - 128)
    x = min(int(max_pos[1]), diff.shape[1] - 128)
    return image[y:y+128, x:x+128], image_ref[y:y+128, x:x+128]


def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def contrast_normalize(image):
    return (image - np.mean(image)) / np.std(image)


def getShift(image, image_ref):
    step = 4
    image = normalize_image(image)
    image_ref = normalize_image(image_ref)
    windows = view_as_windows(image, (512, 512), step)
    diff = np.abs(image - image_ref)
    max_pos = np.where(diff == diff.max())
    y = min(int(max_pos[0]), diff.shape[0] - 512)
    x = min(int(max_pos[1]), diff.shape[1] - 512)
    window_ref = image_ref[y:y + 512, x:x + 512]
    shifts = []
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            shifts += [(i, j, correlation_coefficient(windows[i, j], window_ref))]
    shifts = np.array(shifts)
    shift = shifts[shifts[:, 2] == np.max(shifts[:, 2])][0]
    print(shift)
    plt.subplot(121)
    plt.imshow(windows[int(shift[0]), int(shift[1])], cmap='gray')
    plt.subplot(122)
    plt.imshow(window_ref, cmap='gray')
    plt.savefig('/gss/r.jarolim/iti/hmi_hinode_v12/comparison/window.jpg')
    plt.close()
    return y - shift[0] * step, x - shift[1] * step

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product



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


params = {'axes.labelsize': 'large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
pylab.rcParams.update(params)

for hmi_map, hinode_map, iti_map in zip(hmi_maps, hinode_maps, iti_maps):
    hmi_map = hmi_map.rotate(recenter=True, missing=-1)
    #
    submap = hmi_map.submap(hinode_map.bottom_left_coord, hinode_map.top_right_coord)
    submap = submap.resample(hinode_map.data.shape * u.pix)
    #
    plt.subplot(121)
    plt.imshow(submap.data.astype(np.float32), cmap='gray')
    plt.subplot(122)
    plt.imshow(hinode_map.data.astype(np.float32), cmap='gray')
    plt.savefig('/gss/r.jarolim/iti/hmi_hinode_v12/comparison/img_comparison.jpg')
    plt.close()
    #
    shift = getShift(submap.data.astype(np.float32), hinode_map.data.astype(np.float32))
    print('Shift:', shift)
    hinode_map = Map(hinode_map.data, hinode_map.meta.copy())
    hinode_map.meta['crpix1'] += shift[1]
    hinode_map.meta['crpix2'] += shift[0]
    #
    iti_img = iti_map.submap(hinode_map.bottom_left_coord, hinode_map.top_right_coord).data.astype(np.float32)
    hmi_img = hmi_map.submap(hinode_map.bottom_left_coord, hinode_map.top_right_coord).resample(iti_img.data.shape * u.pix).data.astype(np.float32)
    hinode_img = hinode_map.resample(iti_img.data.shape * u.pix).data.astype(np.float32)
    #
    hmi_difference = np.abs(hmi_img - hinode_img)
    hmi_ssim = structural_similarity(hmi_img, hinode_img)
    # #
    iti_difference = np.abs(iti_img - hinode_img)
    iti_ssim = structural_similarity(iti_img, hinode_img)
    #
    vmin = hinode_map.min()
    vmax = hinode_map.max()
    date = hinode_map.date.datetime
    #
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.5), sharex=True, sharey=True)
    hmi_map.plot(axes=axs[0], cmap='gray', title=date.isoformat(' ', timespec='minutes'))
    axs[0].set_ylabel('Helioprojective Latitude [arcsec]')
    axs[0].set_xlabel('Helioprojective Longitude [arcsec]')
    iti_map.plot(axes=axs[1], cmap='gray', title='')
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
    plt.savefig(os.path.join(evaluation_path, 'hinode_comparison_%s.jpg' % date.isoformat('T', timespec='minutes')),
                dpi=300)
    plt.close()
    print('Diff HMI; SSIM: %.03f; MSE: %.03f' % (hmi_ssim, np.mean(hmi_difference)))
    print('Diff ITI; SSIM: %.03f; MSE: %.03f' % (iti_ssim, np.mean(iti_difference)))
