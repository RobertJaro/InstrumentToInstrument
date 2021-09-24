import glob
import os
from datetime import timedelta, datetime

from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch
from dateutil.parser import parse
from skimage.io import imsave

from iti.data.editor import sdo_norms
from sunpy.map import Map, all_coordinates_from_map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from iti.prediction.translate import STEREOToSDOMagnetogram


from astropy import units as u

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from matplotlib import pyplot as plt

from iti.data.dataset import SOHODataset, STEREODataset

import numpy as np

base_path = "/gss/r.jarolim/iti/stereo_mag_v11"
prediction_path = os.path.join(base_path, 'evaluation')
os.makedirs(prediction_path, exist_ok=True)

soho_basenames = np.array([os.path.basename(f) for f in sorted(glob.glob('/gss/r.jarolim/data/soho_iti2021_prep/171/*.fits'))])
soho_dates = np.array([parse(bn.split('.')[0]) for bn in soho_basenames])
stereo_basenames = np.array(
    [os.path.basename(f) for f in sorted(glob.glob('/gss/r.jarolim/data/stereo_iti2021_prep/171/*.fits'))])
stereo_dates = np.array([parse(bn.split('.')[0]) for bn in stereo_basenames])

min_diff = np.array([np.min(np.abs(stereo_dates - soho_date)) for soho_date in soho_dates])
# filter time diff
cond = (min_diff < timedelta(hours=2)) & (soho_dates < datetime(2007, 6, 1))
soho_dates = soho_dates[cond]
soho_basenames = soho_basenames[cond]
# select corresponding stereo files
stereo_basenames = stereo_basenames[[np.argmin(np.abs(stereo_dates - soho_date)) for soho_date in soho_dates]]

stereo_dataset = [[Map("/gss/r.jarolim/data/stereo_iti2021_prep/%s/%s" % (wl, bn)) for wl in ['171', '195', '284', '304']] for bn in stereo_basenames]

soho_dataset = [[Map("/gss/r.jarolim/data/soho_iti2021_prep/%s/%s" % (wl, bn)) for wl in ['171', '195', '284', '304', 'mag']] for bn in soho_basenames]
translator = STEREOToSDOMagnetogram(model_path=os.path.join(base_path, 'generator_AB.pt'))

result = translator.translate("/gss/r.jarolim/data/stereo_iti2021_prep", basenames=stereo_basenames)

sdo_cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    plt.get_cmap('gray')
]

def getMeanMagneticFlux(s_map):
    s_map = s_map.rotate(recenter=True)
    r_arcsec = s_map.rsun_obs
    bl = SkyCoord(-r_arcsec, -r_arcsec , frame=s_map.coordinate_frame)
    tr = SkyCoord(r_arcsec, r_arcsec, frame=s_map.coordinate_frame)
    s_map = s_map.submap(bl, tr)
    return np.nanmean(np.abs(s_map.data))


mean_mag_iti = []
mean_mag_soho = []
dates = []

for date, (iti_maps, stereo_img, iti_img), soho_map_cube, stereo_map_cube in tqdm(zip(soho_dates, result, soho_dataset, stereo_dataset), total=len(stereo_basenames)):
    # print('Quality', soho_map_cube[-1].meta['quality'])
    # print('Level', soho_map_cube[-1].meta['FDSCALEQ'])
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    for c in range(4):
        axs[0, c].imshow(stereo_img[c], cmap=sdo_cmaps[c])

    for c in range(5):
        s_map = iti_maps[c]
        if c == 4:
            axs[1, c].imshow(np.abs(s_map.data), cmap=cm.hmimag, vmin=-1500, vmax=1500)
        else:
            axs[1, c].imshow(s_map.data, cmap=sdo_cmaps[c], norm=sdo_norms[s_map.wavelength.value])

    for c in range(5):
        s_map = soho_map_cube[c]
        s_map = s_map.rotate(recenter=True)
        bl = SkyCoord(iti_maps[c].bottom_left_coord.Tx, iti_maps[c].bottom_left_coord.Ty, frame=s_map.coordinate_frame)
        tr = SkyCoord(iti_maps[c].top_right_coord.Tx, iti_maps[c].top_right_coord.Ty, frame=s_map.coordinate_frame)
        s_map = s_map.submap(bl, tr)
        if c == 4:
            hpc_coords = all_coordinates_from_map(s_map)
            r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
            s_map.data[r > 1] = np.nan
            axs[2, c].imshow(np.abs(s_map.data), cmap=cm.hmimag, vmin=-1500, vmax=1500)
        else:
            axs[2, c].imshow(s_map.data, cmap=sdo_cmaps[c], norm=ImageNormalize(stretch=AsinhStretch(0.005)))

    [ax.set_axis_off() for ax in np.ravel(axs)]

    plt.tight_layout(0.1)
    plt.savefig(os.path.join(base_path, 'evaluation/%s.jpg' % date.isoformat('T')), dpi=300)
    plt.close()

    dates.append(date)
    mean_mag_iti.append(getMeanMagneticFlux(iti_maps[-1]))
    mean_mag_soho.append(getMeanMagneticFlux(soho_map_cube[-1]))

    plt.figure(figsize=(8, 4))
    plt.plot(dates, mean_mag_iti, '-o', label='ITI')
    plt.plot(dates, mean_mag_soho, '-o', label='SOHO')
    plt.legend()
    plt.savefig(os.path.join(base_path, 'evaluation/mag_comparison.jpg'), dpi=300)
    plt.close()

    s_map = iti_maps[-1]
    r_arcsec = s_map.rsun_obs
    bl = SkyCoord(-r_arcsec, -r_arcsec, frame=s_map.coordinate_frame)
    tr = SkyCoord(r_arcsec, r_arcsec, frame=s_map.coordinate_frame)
    s_map = s_map.submap(bl, tr)
    hpc_coords = all_coordinates_from_map(s_map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
    data = s_map.data
    data = sdo_cmaps[-1](ImageNormalize(vmin=-1000, vmax=1000)(data))[..., :-1]
    data[r > 1] = [0, 0, 0]
    imsave(os.path.join(base_path, 'evaluation/%s_iti.jpg' % date.isoformat('T')), data, check_contrast=False)

    s_map = soho_map_cube[-1]
    s_map = s_map.rotate(recenter=True)
    r_arcsec = s_map.rsun_obs
    bl = SkyCoord(-r_arcsec, -r_arcsec, frame=s_map.coordinate_frame)
    tr = SkyCoord(r_arcsec, r_arcsec, frame=s_map.coordinate_frame)
    s_map = s_map.submap(bl, tr)
    hpc_coords = all_coordinates_from_map(s_map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
    data = s_map.data
    data = sdo_cmaps[-1](ImageNormalize(vmin=-1000, vmax=1000)(np.abs(data)))[..., :-1]
    data[r > 1] = [0,0,0]
    imsave(os.path.join(base_path, 'evaluation/%s_soho.jpg' % date.isoformat('T')), data, check_contrast=False)

    for s_map, cmap, id in zip(stereo_map_cube, sdo_cmaps, ['171', '195', '284', '304']):
        s_map = s_map.rotate(recenter=True)
        r_arcsec = s_map.rsun_obs
        bl = SkyCoord(-r_arcsec, -r_arcsec, frame=s_map.coordinate_frame)
        tr = SkyCoord(r_arcsec, r_arcsec, frame=s_map.coordinate_frame)
        s_map = s_map.submap(bl, tr)
        data = s_map.data
        data = cmap(ImageNormalize(stretch=AsinhStretch(0.005))(np.abs(data)))[..., :-1]
        imsave(os.path.join(base_path, 'evaluation/%s_%s.jpg' % (date.isoformat('T'), id)), data, check_contrast=False)