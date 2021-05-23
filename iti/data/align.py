from warnings import simplefilter

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import block_reduce
from skimage.util import view_as_windows
from sunpy.map import Map

from matplotlib import pyplot as plt

from astropy import units as u

def alignMaps(align_map, ref_map):
    simplefilter('ignore')
    align_map = Map(align_map.data, align_map.meta.copy())
    # adjust scale
    #
    width = align_map.bottom_left_coord.Tx - align_map.top_right_coord.Tx
    height = align_map.bottom_left_coord.Ty - align_map.top_right_coord.Ty
    #v
    for fov, reduction_scale in zip([2, 1, 0.7, 0.51], [32, 8, 4, 2]):
        # initial align
        coord = align_map.center
        bl = SkyCoord(coord.Tx - width * fov, coord.Ty - height * fov, frame=ref_map.coordinate_frame)
        tr = SkyCoord(coord.Tx + width * fov, coord.Ty + height * fov, frame=ref_map.coordinate_frame)
        submap = ref_map.submap(bottom_left=bl, top_right=tr)
        #
        shift = getShift(align_map.data.astype(np.float32), submap.data.astype(np.float32), reduction_block=(reduction_scale, reduction_scale))
        print(shift)
        align_map.meta['crpix1'] += shift[1]
        align_map.meta['crpix2'] += shift[0]
    #
    plt.imshow(align_map.data.astype(np.float32), cmap='gray')
    plt.savefig('/gss/r.jarolim/iti/hmi_hinode_v12/compare/align.jpg')
    plt.close()
    coord = align_map.center
    bl = SkyCoord(coord.Tx - width / 2, coord.Ty - height / 2, frame=ref_map.coordinate_frame)
    tr = SkyCoord(coord.Tx + width / 2, coord.Ty + height / 2, frame=ref_map.coordinate_frame)
    submap = ref_map.submap(bottom_left=bl, top_right=tr)
    plt.imshow(submap.data.astype(np.float32), cmap='gray')
    plt.savefig('/gss/r.jarolim/iti/hmi_hinode_v12/compare/ref.jpg')
    plt.close()
    return align_map


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
    best_shift = shifts[shifts[:, 2] == np.max(shifts[:, 2])][0]
    #
    plt.imshow(image, cmap='gray')
    plt.savefig('/gss/r.jarolim/iti/hmi_hinode_v12/compare/match_ref_%d.jpg' % reduction_block[0])
    plt.imshow(windows[int(best_shift[0]), int(best_shift[1])], cmap='gray')
    plt.savefig('/gss/r.jarolim/iti/hmi_hinode_v12/compare/match_%d.jpg' % reduction_block[0])
    plt.close()
    #
    center = (image_ref.shape[0] // 2 - image.shape[0] // 2, image_ref.shape[1] // 2 - image.shape[1] // 2)
    return (center[0] - best_shift[0]) * reduction_block[0], (center[1] - best_shift[1]) * reduction_block[1]


def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product