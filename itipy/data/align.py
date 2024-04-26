from warnings import simplefilter

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import block_reduce
from skimage.util import view_as_windows
from sunpy.map import Map


def alignMaps(align_map, ref_map):
    simplefilter('ignore')
    original_map = align_map
    # downscale to reference map
    scale_factor = align_map.scale[0] / ref_map.scale[0]
    align_map = Map(align_map.data, align_map.meta.copy())
    align_map = align_map.resample([int(align_map.data.shape[1] * scale_factor),
                                    int(align_map.data.shape[0] * scale_factor)] * u.pix)
    # adjust scale
    #
    width = align_map.bottom_left_coord.Tx - align_map.top_right_coord.Tx
    height = align_map.bottom_left_coord.Ty - align_map.top_right_coord.Ty
    #
    for fov, reduction_scale in zip([1.5, 1, 0.7, 0.51], [8, 4, 2, 1]):
        # initial align
        coord = align_map.center
        bl = SkyCoord(coord.Tx - width * fov, coord.Ty - height * fov, frame=ref_map.coordinate_frame)
        tr = SkyCoord(coord.Tx + width * fov, coord.Ty + height * fov, frame=ref_map.coordinate_frame)
        submap = ref_map.submap(bottom_left=bl, top_right=tr)
        #
        shift = getShift(align_map.data.astype(np.float32), submap.data.astype(np.float32),
                         reduction_block=(reduction_scale, reduction_scale))
        # apply shift
        align_map.meta['crpix1'] += shift[1]
        align_map.meta['crpix2'] += shift[0]
    # copy shift information to original image size
    new_meta = original_map.meta.copy()
    lon, lat = original_map._get_lon_lat(align_map.center.frame)
    new_meta['crpix1'] = (original_map.dimensions[0].value + 1) / 2.
    new_meta['crpix2'] = (original_map.dimensions[1].value + 1) / 2.
    new_meta['crval1'] = lon.value
    new_meta['crval2'] = lat.value
    #
    return Map(original_map.data, new_meta)


def getShift(image, image_ref, reduction_block=(1, 1)):
    """
    Pixel shift between two images by cross-correlation of the subframes.
    :param image: smaller image patch to align with the reference image
    :param image_ref: reference image
    :return: pixel shift between the image to the reference image
    """
    image = block_reduce(image, reduction_block, func=np.mean)
    image_ref = block_reduce(image_ref, reduction_block, func=np.mean)
    windows = view_as_windows(image_ref, image.shape, 1)
    #
    shifts = []
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            shifts += [(i, j, correlation_coefficient(windows[i, j], image))]
    shifts = np.array(shifts)
    best_shift = shifts[np.argmax(shifts[:, 2])]
    #
    center = (image_ref.shape[0] // 2 - image.shape[0] // 2, image_ref.shape[1] // 2 - image.shape[1] // 2)
    return (center[0] - best_shift[0]) * reduction_block[0], (center[1] - best_shift[1]) * reduction_block[1]


def correlation_coefficient(patch1, patch2):
    product = np.nanmean((patch1 - np.nanmean(patch1)) * (patch2 - np.nanmean(patch2)))
    stds = np.nanstd(patch1) * np.nanstd(patch2)
    if stds == 0:
        return 0
    else:
        product /= stds
        return product
