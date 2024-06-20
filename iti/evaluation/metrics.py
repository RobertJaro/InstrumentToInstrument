import numpy as np
from skimage.metrics import structural_similarity


def calibrate(data, ref_data):
    mean_ref = ref_data.mean()
    std_ref = ref_data.std()

    mean_data = data.mean()
    std_data = data.std()

    return (data - mean_data) * (std_ref / std_data) + mean_ref


def rms_contrast(data, axis=(-2, -1)):
    return np.sqrt(np.nanmean((data - np.nanmean(data, axis=axis)) ** 2, axis=axis))

def rms_contrast_diff(img, img_ref, axis=(-2, -1)):
    img_rmsc = rms_contrast(img, axis=axis)
    img_ref_rmsc = rms_contrast(img_ref, axis=axis)
    return np.abs(img_rmsc - img_ref_rmsc)


def psnr(img, img_ref, axis=(-2, -1), data_range=1.0):
    # data_range = np.nanmax(img_ref, axis=axis) - np.nanmin(img_ref, axis=axis)
    mse = np.nanmean((img - img_ref) ** 2, axis=axis)
    return 10 * np.log10((data_range ** 2) / mse)

def mae(img, img_ref, axis=(-2, -1)):
    return np.nanmean(np.abs(img - img_ref), axis=axis)

def image_correlation(img, img_ref, axis=(-2, -1)):
    img_mean = np.nanmean(img, axis=axis, keepdims=True)
    img_ref_mean = np.nanmean(img_ref, axis=axis, keepdims=True)
    img_diff = img - img_mean
    img_ref_diff = img_ref - img_ref_mean
    return np.nanmean(img_diff * img_ref_diff, axis=axis) / (np.nanstd(img, axis=axis) * np.nanstd(img_ref, axis=axis))

def normalize(img):
    img = img - np.nanmin(img)
    img = img / np.nanmax(img)
    return img

def ssim(img, img_ref, data_range=1.0):
    ssim = structural_similarity(img, img_ref, data_range=data_range)
    return ssim