import glob
import os

from astropy.coordinates import SkyCoord
from sunpy.map import Map
from sunpy.physics.differential_rotation import solar_rotate_coordinate

from iti.data.align import alignMaps
from iti.translate import HMIToHinode



import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from astropy import units as u

hmi_shape = 4096
patch_shape = 1024
n_patches = hmi_shape // patch_shape
base_path = '/gss/r.jarolim/iti/hmi_hinode_v12'
evaluation_path = os.path.join(base_path, "video_2014_11_22")
zoom_in_path = os.path.join(evaluation_path, "zoom_in")
series_path = os.path.join(evaluation_path, "series")
result_path = os.path.join(evaluation_path, "result")
os.makedirs(zoom_in_path, exist_ok=True)
os.makedirs(series_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

hinode_sample = '/gss/r.jarolim/data/hinode/level1/FG20141122_010538.9.fits' #'/gss/r.jarolim/data/hinode/level1/FG20131118_174620.2.fits'

translator = HMIToHinode(model_path=os.path.join(base_path, 'generator_AB.pt'), patch_factor=3)
hmi_files = sorted(glob.glob('/gss/r.jarolim/data/hmi_video_2014_11_22/6173/*.fits'))
filtered_hmi_files = [f for f in hmi_files if not
os.path.exists(os.path.join(result_path, '%s.fits' % Map(f).date.datetime.isoformat('T')))]

# convert iti
for iti_map in tqdm(translator.translate(filtered_hmi_files), total=len(filtered_hmi_files)):
    iti_map.save(os.path.join(result_path, '%s.fits' % iti_map.date.datetime.isoformat('T')))

# load hinode
hinode_map = Map(hinode_sample)
bl, tr = hinode_map.bottom_left_coord, hinode_map.top_right_coord
hinode_map = hinode_map.rotate(scale=hinode_map.scale[0] / (0.15 * u.arcsec / u.pix), missing=np.nan)
hinode_map = hinode_map.submap(bl, tr)
hinode_map = alignMaps(hinode_map, Map(hmi_files[-1]).rotate(recenter=True))
hinode_center_coord = hinode_map.center
# load iti
iti_files = sorted(glob.glob(os.path.join(result_path, '*.fits')))

iti_ref = Map(iti_files[0]).rotate(recenter=True)
n_samples = len(iti_files)
lin_x = np.linspace(iti_ref.center.Tx.value, hinode_center_coord.Tx.value, n_samples * 2 // 3).tolist()
lin_y = np.linspace(iti_ref.center.Ty.value, hinode_center_coord.Ty.value, n_samples * 2 // 3).tolist()
lin_w = np.linspace(iti_ref.top_right_coord.Tx.value - iti_ref.bottom_left_coord.Tx.value,
                    hinode_map.top_right_coord.Tx.value - hinode_map.bottom_left_coord.Tx.value, n_samples * 2 // 3).tolist()
# use longer axis for height
lin_h = np.linspace(iti_ref.top_right_coord.Tx.value - iti_ref.bottom_left_coord.Tx.value,
                    hinode_map.top_right_coord.Tx.value - hinode_map.bottom_left_coord.Tx.value, n_samples * 2 // 3).tolist()

lin_x += [lin_x[-1]] * round(n_samples / 3)
lin_y += [lin_y[-1]] * round(n_samples / 3)
lin_w += [lin_w[-1]] * round(n_samples / 3)
lin_h += [lin_h[-1]] * round(n_samples / 3)

for path, x, y, w, h in zip(iti_files, lin_x, lin_y, lin_w, lin_h):
    s_map = Map(path)
    #
    center_coord = SkyCoord(x * u.arcsec, y * u.arcsec, frame=hinode_map.coordinate_frame)
    rotated = solar_rotate_coordinate(center_coord, observer=s_map.observer_coordinate)
    center_coord = center_coord if np.isnan(rotated.Tx.value) else rotated
    #
    bl_coord = SkyCoord(center_coord.Tx - w / 2 * u.arcsec, center_coord.Ty - h / 2 * u.arcsec,
                        frame=s_map.coordinate_frame)
    tr_coord = SkyCoord(center_coord.Tx + w / 2 * u.arcsec, center_coord.Ty + h / 2 * u.arcsec,
                        frame=s_map.coordinate_frame)
    #
    sub_map = s_map.submap(bottom_left=bl_coord, top_right=tr_coord)
    plt.figure(figsize=(7, 6))
    plt.subplot(111, projection=sub_map)
    plt.imshow(sub_map.data, cmap=sub_map.plot_settings['cmap'], vmin=3742, vmax=33737)
    plt.title(sub_map.date.datetime.isoformat(' ', timespec='seconds'), fontsize=18)
    plt.ylabel('Helioprojective Latitude [arcsec]', fontsize=16)
    plt.xlabel('Helioprojective Longitude [arcsec]', fontsize=16)
    plt.savefig(os.path.join(zoom_in_path, '%s.jpg' % s_map.date.to_datetime().isoformat('T')), dpi=150)
    plt.close()

# hinode_files = sorted(glob.glob('/gss/r.jarolim/data/hinode_video_2014_11_22/*.fits'))
# hmi_dates = [parse(os.path.basename(f).split('.')[0]) for f in hmi_files]
# hinode_dates = np.array([parse(os.path.basename(f).split('.')[0][2:].replace('_', 'T')) for f in hinode_files])
# hinode_files = np.array(hinode_files)[[np.argmin(np.abs(d - hinode_dates)) for d in hmi_dates]]

iti_maps = (Map(f) for f in iti_files)
hmi_maps = (Map(f).rotate(recenter=True) for f in hmi_files)
# hinode_maps = (Map(f) for f in hinode_files)


w = hinode_map.top_right_coord.Tx.value - hinode_map.bottom_left_coord.Tx.value
h = hinode_map.top_right_coord.Tx.value - hinode_map.bottom_left_coord.Tx.value
hinode_center_coord = hinode_map.center

for iti_map, hmi_map in zip(iti_maps, hmi_maps):
    center_coord = solar_rotate_coordinate(hinode_center_coord, observer=iti_map.observer_coordinate)
    bl_coord = SkyCoord(center_coord.Tx - w / 2 * u.arcsec, center_coord.Ty - h / 2 * u.arcsec,
                        frame=iti_map.coordinate_frame)
    tr_coord = SkyCoord(center_coord.Tx + w / 2 * u.arcsec, center_coord.Ty + h / 2 * u.arcsec,
                        frame=iti_map.coordinate_frame)
    #
    iti_sub_map = iti_map.submap(bottom_left=bl_coord, top_right=tr_coord)
    hmi_sub_map = hmi_map.submap(bottom_left=bl_coord, top_right=tr_coord)
    # hinode_sub_map = hinode_map.rotate().submap(bottom_left=bl_coord, top_right=tr_coord)
    plt.figure(figsize=(12, 6))
    plt.subplot(121, projection=hmi_sub_map)
    plt.imshow(hmi_sub_map.data, cmap=hmi_sub_map.plot_settings['cmap'])
    plt.ylabel(' ')
    plt.xlabel(' ')
    plt.title('HMI (%s)' % iti_map.date.to_datetime().isoformat(' ', timespec='seconds'), fontsize=24)
    # plt.ylabel('Helioprojective Latitude [arcsec]', fontsize=20)
    # plt.xlabel('Helioprojective Longitude [arcsec]', fontsize=20)
    plt.subplot(122, projection=iti_sub_map)
    plt.imshow(iti_sub_map.data, cmap=iti_sub_map.plot_settings['cmap'])
    plt.ylabel(' ')
    plt.xlabel(' ')
    plt.title('ITI', fontsize=24)
    # plt.xlabel('Helioprojective Longitude [arcsec]', fontsize=20)
    # plt.subplot(133, projection=iti_sub_map)
    # plt.imshow(hinode_sub_map.data, cmap=iti_sub_map.plot_settings['cmap'])
    # plt.ylabel(' ')
    # plt.xlabel(' ')
    # plt.xlabel('Helioprojective Longitude [arcsec]', fontsize=20)
    # plt.title('Hinode (%s)' % hinode_map.date.to_datetime().isoformat(' ', timespec='seconds'), fontsize=24)
    # plt.tight_layout(pad=2)
    plt.savefig(os.path.join(series_path, '%s.jpg' % iti_map.date.to_datetime().isoformat('T')), dpi=150)
    plt.close()

# hmi_sub_map = hmi_maps[-1].submap(hinode_map.bottom_left_coord, hinode_map.top_right_coord)
# iti_sub_map = Map(iti_files[-1]).rotate(recenter=True).submap(hinode_map.bottom_left_coord, hinode_map.top_right_coord)
#
# imsave(evaluation_path + '/hmi.jpg', hmi_sub_map.data, check_contrast=False)
# imsave(evaluation_path + '/iti.jpg', iti_sub_map.data, check_contrast=False)
# imsave(evaluation_path + '/hinode.jpg', hinode_map.data, check_contrast=False)