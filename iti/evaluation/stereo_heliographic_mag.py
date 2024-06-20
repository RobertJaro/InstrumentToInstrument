import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
import sunpy.sun
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from datetime import datetime
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
from sunpy.coordinates import frames
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from iti.data.editor import AIAPrepEditor, sdo_norms
from iti.translate import STEREOToSDOMagnetogram

wavelengths = [(171, 171), (195, 193, ), (284, 211, ), (304, 304, )]
cmaps = [cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia304]

aia_prep_editor = AIAPrepEditor()

def build_map(maps, out_wcs, header, shape_out):
    coordinates = tuple(map(sunpy.map.all_coordinates_from_map, maps))
    weights = [
        np.abs(coord.transform_to(frames.HeliographicStonyhurst).lon.value - s_map.observer_coordinate.lon.value) < 45
        for coord, s_map in zip(coordinates, maps)]
    weights = [*weights[:2], weights[2] * 100]
    array, footprint = reproject_and_coadd(maps, out_wcs, shape_out, input_weights=weights,
                                           reproject_function=reproject_interp, match_background=False)
    outmap = sunpy.map.Map((array, header))
    outmap.plot_settings = maps[-1].plot_settings
    return outmap


def format_axis(ax):
    lon, lat = ax.coords
    lon.set_coord_type("longitude")
    lon.coord_wrap = 180
    lon.set_format_unit(u.deg)
    lat.set_coord_type("latitude")
    lat.set_format_unit(u.deg)
    lon.set_axislabel('Heliographic Longitude', fontsize=18)
    lat.set_axislabel('Heliographic Latitude', fontsize=18)
    lon.set_ticks(spacing=30 * u.deg, color='k')
    lat.set_ticks(spacing=30 * u.deg, color='k')
    scale = shape_out[0] / 180
    _ = ax.axis((shape_out[1] / 2 - scale * 120, shape_out[1] / 2 + scale * 120,
                 shape_out[0] / 2 - scale * 60, shape_out[0] / 2 + scale * 60))
    ax.axvline(shape_out[1] / 2 - scale * 45, color='red', linestyle='--')
    ax.axvline(shape_out[1] / 2 + scale * 45, color='red', linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=14)

base = '/gpfs/gpfs0/robert.jarolim/data/iti/stereo_heliographic_prep'
base_path = "/gpfs/gpfs0/robert.jarolim/iti/stereo_to_sdo_mag_v1"#"/gss/r.jarolim/iti/stereo_v7"
prediction_path = os.path.join(base_path, 'evaluation')
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = STEREOToSDOMagnetogram(model_path=os.path.join(base_path, 'generator_AB.pt'))

plot_euv = True

for j in tqdm([3, 9, 16]):
    date = datetime(2010, 12, j)
    id = date.isoformat('T')

    iti_A_maps = next(translator.translate(base, basenames=['%s_A.fits' % id]))
    iti_B_maps = next(translator.translate(base, basenames=['%s_B.fits' % id]))
    iti_A_map, iti_B_map = iti_A_maps[-1], iti_B_maps[-1]

    hmi_map = Map(base + '/mag/%s_sdo.fits' % id)
    hmi_map = hmi_map.resample(iti_A_map.data.shape * u.pix)
    hmi_map.meta['rsun_ref'] = sunpy.sun.constants.radius.to_value(u.m)
    #
    shape_out = (1024, 2048)
    header = sunpy.map.make_fitswcs_header(shape_out,
                                           SkyCoord(0, 0, unit=u.deg,
                                                    frame="heliographic_stonyhurst",
                                                    obstime=hmi_map.date),
                                           scale=[180 / shape_out[0],
                                                  360 / shape_out[1]] * u.deg / u.pix,
                                           wavelength=int(hmi_map.meta['wavelnth']) * u.AA,
                                           projection_code="CAR")
    out_wcs = WCS(header)
    # create maps
    full_iti_map = build_map([iti_A_map, iti_B_map, hmi_map], out_wcs, header, shape_out)
    # plot
    plt.figure(figsize=(15, 9))
    ax = plt.subplot(projection=out_wcs)
    ax.imshow(np.abs(full_iti_map.data), vmin=-1000, vmax=1000, cmap='gray')
    format_axis(ax)
    ax.set_title(date.isoformat(' ', timespec='hours'), fontsize=24)
    #plt.tight_layout(4)
    #
    plt.savefig(os.path.join(prediction_path, 'heliographic_map_%s.jpg' % date.isoformat('T')), dpi=300, transparent=False)
    plt.close()
    #
    if not plot_euv:
        continue
    file_list = [(base + '/171/%s_A.fits' % id, base + '/171/%s_B.fits' % id, base + '/171/%s_sdo.fits' % id),
                 (base + '/195/%s_A.fits' % id, base + '/195/%s_B.fits' % id, base + '/195/%s_sdo.fits' % id),
                 (base + '/284/%s_A.fits' % id, base + '/284/%s_B.fits' % id, base + '/284/%s_sdo.fits' % id),
                 (base + '/304/%s_A.fits' % id, base + '/304/%s_B.fits' % id, base + '/304/%s_sdo.fits' % id), ]
    stereo_A_maps = [Map(files[0]) for files in file_list]
    stereo_B_maps = [Map(files[1]) for files in file_list]
    sdo_maps = [Map(files[2]) for files in file_list]
    for i in tqdm([0, 1 ,2, 3], desc='Building Maps'):
        stereo_A_map, stereo_B_map, iti_A_map, iti_B_map, sdo_map = stereo_A_maps[i], stereo_B_maps[i], iti_A_maps[i], \
                                                                    iti_B_maps[i], sdo_maps[i]
        #
        sdo_map = aia_prep_editor.call(sdo_map)
        sdo_map.meta['rsun_ref'] = sunpy.sun.constants.radius.to_value(u.m)
        #
        shape_out = (1024, 2048)
        header = sunpy.map.make_fitswcs_header(shape_out,
                                               SkyCoord(0, 0, unit=u.deg,
                                                        frame="heliographic_stonyhurst",
                                                        obstime=sdo_map.date),
                                               scale=[180 / shape_out[0],
                                                      360 / shape_out[1]] * u.deg / u.pix,
                                               wavelength=int(sdo_map.meta['wavelnth']) * u.AA,
                                               projection_code="CAR")
        out_wcs = WCS(header)
        #
        # create map
        full_iti_map = build_map([iti_A_map, iti_B_map, sdo_map], out_wcs, header, shape_out)
        # plot
        plt.figure(figsize=(15, 9))
        ax = plt.subplot(projection=out_wcs)
        ax.imshow(full_iti_map.data, norm=sdo_norms[sdo_map.wavelength.value], cmap=cmaps[i])
        format_axis(ax)
        ax.set_title(date.isoformat(' ', timespec='hours'), fontsize=24)
        plt.savefig(os.path.join(prediction_path,
                                 'heliographic_map_%d_%s_iti.jpg' % (sdo_map.wavelength.value, date.isoformat('T'), )),
                    dpi=300, transparent=False)
        plt.close()