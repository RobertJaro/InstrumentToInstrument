import logging
import os
import random
import shutil
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from random import randint

import astropy.io.ascii
import numpy as np
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.util import get_correction_table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, AsinhStretch
from dateutil.parser import parse
from matplotlib.colors import LogNorm, Normalize
from scipy import ndimage
from skimage.transform import pyramid_reduce
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map, header_helper
from matplotlib import pyplot as plt

class Editor(ABC):

    def convert(self, data, **kwargs):
        result = self.call(data, **kwargs)
        if isinstance(result, tuple):
            data, add_kwargs = result
            kwargs.update(add_kwargs)
        else:
            data = result
        return data, kwargs

    @abstractmethod
    def call(self, data, **kwargs):
        raise NotImplementedError()


sdo_norms = {94: ImageNormalize(vmin=0, vmax=411, stretch=AsinhStretch(0.005), clip=True),
             131: ImageNormalize(vmin=0, vmax=1268, stretch=AsinhStretch(0.005), clip=True),
             171: ImageNormalize(vmin=0, vmax=6463, stretch=AsinhStretch(0.005), clip=True),
             193: ImageNormalize(vmin=0, vmax=8392, stretch=AsinhStretch(0.005), clip=True),
             211: ImageNormalize(vmin=0, vmax=4184, stretch=AsinhStretch(0.005), clip=True),
             304: ImageNormalize(vmin=0, vmax=6481, stretch=AsinhStretch(0.005), clip=True),
             335: ImageNormalize(vmin=0, vmax=637, stretch=AsinhStretch(0.005), clip=True),
             1600: ImageNormalize(vmin=0, vmax=4000, stretch=AsinhStretch(0.005), clip=True),  # TODO
             1700: ImageNormalize(vmin=0, vmax=4000, stretch=AsinhStretch(0.005), clip=True),  # TODO
             'mag': ImageNormalize(vmin=-1000, vmax=1000, stretch=LinearStretch(), clip=True),
             'continuum': ImageNormalize(vmin=0, vmax=70000, stretch=LinearStretch(), clip=True),
             }

soho_norms = {171: ImageNormalize(vmin=0, vmax=10394, stretch=AsinhStretch(0.005), clip=True),
              195: ImageNormalize(vmin=0, vmax=7609, stretch=AsinhStretch(0.005), clip=True),
              284: ImageNormalize(vmin=0, vmax=1772, stretch=AsinhStretch(0.005), clip=True),
              304: ImageNormalize(vmin=0, vmax=8252, stretch=AsinhStretch(0.005), clip=True),
              6173: ImageNormalize(vmin=-1000, vmax=1000, stretch=LinearStretch(), clip=True),
              }

secchi_norms = {171: ImageNormalize(vmin=0, vmax=11523, stretch=AsinhStretch(0.005), clip=True),
                195: ImageNormalize(vmin=0, vmax=6768, stretch=AsinhStretch(0.005), clip=True),
                284: ImageNormalize(vmin=0, vmax=1927, stretch=AsinhStretch(0.005), clip=True),
                304: ImageNormalize(vmin=0, vmax=8378, stretch=AsinhStretch(0.005), clip=True),
                }

hinode_norm = {'continuum': ImageNormalize(vmin=0, vmax=50000, stretch=LinearStretch(), clip=True)}


class LoadFITSEditor(Editor):

    def call(self, map_path, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            dst = shutil.copy(map_path, os.path.join(os.environ.get("TMPDIR"), os.path.basename(map_path)))
            hdul = fits.open(dst)
            os.remove(dst)
            hdul.verify("fix")
            data, header = hdul[0].data, hdul[0].header
            hdul.close()
        return data, {"header": header}


class LoadMapEditor(Editor):

    def call(self, data, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_map = Map(data)
            return s_map, {'path': data}


class SubMapEditor(Editor):

    def __init__(self, coords):
        self.coords = coords

    def call(self, map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            return map.submap(SkyCoord(*self.coords, frame=map.coordinate_frame))


class MapToDataEditor(Editor):

    def call(self, map, **kwargs):
        return map.data, {"header": map.meta}

class DataToMapEditor(Editor):

    def call(self, data, **kwargs):
        return Map(data[0], kwargs['header'])

class ContrastNormalizeEditor(Editor):

    def __init__(self, use_median=False):
        self.use_median = use_median

    def call(self, data, **kwargs):
        shift = np.nanmedian(data) if self.use_median else np.nanmean(data)
        data = (data - shift) / (np.nanstd(data) + 10e-8)
        return data


class ImageNormalizeEditor(Editor):

    def __init__(self, vmin=None, vmax=None, stretch=LinearStretch()):
        self.norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch, clip=True)

    def call(self, data, **kwargs):
        data = self.norm(data).data * 2 - 1
        return data


class NormalizeEditor(Editor):

    def __init__(self, norm):
        self.norm = norm

    def call(self, data, **kwargs):
        data = self.norm(data).data * 2 - 1
        return data


class ReshapeEditor(Editor):

    def __init__(self, shape):
        self.shape = shape

    def call(self, data, **kwargs):
        data = data[:self.shape[1], :self.shape[2]]
        return np.reshape(data, self.shape).astype(np.float32)


class ExpandDimsEditor(Editor):

    def __init__(self, axis=0):
        self.axis = axis

    def call(self, data, **kwargs):
        return np.expand_dims(data, axis=self.axis)


class NanEditor(Editor):
    def __init__(self, nan=0):
        self.nan = nan

    def call(self, data, **kwargs):
        data = np.nan_to_num(data, nan=self.nan)
        return data


class KSOPrepEditor(Editor):
    def __init__(self, add_rotation=False):
        self.add_rotation = add_rotation

        super().__init__()

    def call(self, kso_map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            kso_map.meta["waveunit"] = "ag"
            kso_map.meta["arcs_pp"] = kso_map.scale[0].value
            if 'exptime' not in kso_map.meta:
                kso_map.meta['exptime'] = kso_map.meta['exp_time'] / 1000 # ms to s

            if self.add_rotation:
                angle = -kso_map.meta["angle"]
            else:
                angle = 0
            c = np.cos(np.deg2rad(angle))
            s = np.sin(np.deg2rad(angle))

            kso_map.meta["PC1_1"] = c
            kso_map.meta["PC1_2"] = -s
            kso_map.meta["PC2_1"] = s
            kso_map.meta["PC2_2"] = c

            return kso_map


class KSOFilmPrepEditor(Editor):
    def __init__(self, add_rotation=False):
        self.add_rotation = add_rotation

        super().__init__()

    def call(self, data, **kwargs):
        data = data[0]
        h = kwargs['header']
        coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime=parse(h['DATE_OBS']), observer='earth',
                         frame=frames.Helioprojective)
        header = header_helper.make_fitswcs_header(
            data, coord,
            rotation_angle=h["ANGLE"] * u.deg if self.add_rotation else 0 * u.deg,
            reference_pixel=u.Quantity([h['CENTER_X'], h['CENTER_Y']] * u.pixel),
            scale=u.Quantity([h['CDELT1'], h['CDELT2']] * u.arcsec / u.pix),
            instrument=h["INSTRUME"],
            exposure=h["EXPTIME"] * u.ms,
            wavelength=h["WAVELNTH"] * u.angstrom)

        return Map(data, header)


class AIAPrepEditor(Editor):
    def __init__(self):
        super().__init__()

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")  # ignore warnings
        s_map = correct_degradation(s_map, correction_table=get_local_correction_table())
        data = np.nan_to_num(s_map.data)
        data = data / s_map.meta["exptime"]
        return Map(data.astype(np.float32), s_map.meta)


class NormalizeExposureEditor(Editor):
    def __init__(self, target=1 * u.s):
        self.target = target
        super().__init__()

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")  # ignore warnings
        data = s_map.data
        data = data / s_map.exposure_time.to(u.s).value * self.target.to(u.s).value
        return Map(data.astype(np.float32), s_map.meta)


class NormalizeRadiusEditor(Editor):
    def __init__(self, resolution, padding_factor=0.1, crop=True, **kwargs):
        self.padding_factor = padding_factor
        self.resolution = resolution
        self.crop = crop
        super(NormalizeRadiusEditor, self).__init__(**kwargs)

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")  # ignore warnings
        r_obs_pix = s_map.rsun_obs / s_map.scale[0]  # normalize solar radius
        r_obs_pix = (1 + self.padding_factor) * r_obs_pix
        scale_factor = self.resolution / (2 * r_obs_pix.value)
        s_map = Map(np.nan_to_num(s_map.data).astype(np.float32), s_map.meta)
        s_map = s_map.rotate(recenter=True, scale=scale_factor, missing=0, order=3)
        if self.crop:
            arcs_frame = (self.resolution / 2) * s_map.scale[0].value
            s_map = s_map.submap(SkyCoord([-arcs_frame, arcs_frame] * u.arcsec,
                                          [-arcs_frame, arcs_frame] * u.arcsec,
                                          frame=s_map.coordinate_frame))
            pad_x = s_map.data.shape[0] - self.resolution
            pad_y = s_map.data.shape[1] - self.resolution
            s_map = s_map.submap([pad_x // 2, pad_y // 2] * u.pix,
                                 [pad_x // 2 + self.resolution - 1, pad_y // 2 + self.resolution - 1] * u.pix)
        s_map.meta['r_sun'] = s_map.rsun_obs.value / s_map.meta['cdelt1']
        return s_map


class ScaleEditor(Editor):
    def __init__(self, arcspp):
        self.arcspp = arcspp
        super(ScaleEditor, self).__init__()

    def call(self, s_map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            scale_factor = s_map.scale[0].value / self.arcspp
            new_dimensions = [int(s_map.data.shape[1] * scale_factor),
                              int(s_map.data.shape[0] * scale_factor)] * u.pixel
            s_map = s_map.resample(new_dimensions)
            return Map(s_map.data.astype(np.float32), s_map.meta)


class SubmapSolarRadiiEditor(Editor):
    def __init__(self, solar_radii=1):
        self.solar_radii = solar_radii
        super(SubmapSolarRadiiEditor, self).__init__()

    def call(self, s_map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            arcs_frame = s_map.rsun_obs * self.solar_radii
            s_map = s_map.submap(SkyCoord([-arcs_frame, arcs_frame] * u.arcsec,
                                          [-arcs_frame, arcs_frame] * u.arcsec,
                                          frame=s_map.coordinate_frame))

            return s_map


class PyramidRescaleEditor(Editor):

    def __init__(self, scale=2):
        self.scale = scale

    def call(self, data, **kwargs):
        if self.scale == 1:
            return data
        data = pyramid_reduce(data, downscale=self.scale)
        return data


class LoadNumpyEditor(Editor):

    def call(self, data, **kwargs):
        return np.load(data)


class StackEditor(Editor):

    def __init__(self, data_sets):
        self.data_sets = data_sets

    def call(self, data, **kwargs):
        return np.concatenate([dp[data] for dp in self.data_sets], 0)


class RemoveOffLimbEditor(Editor):

    def __init__(self, fill_value=0):
        self.fill_value = fill_value

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")  # ignore warnings
        hpc_coords = all_coordinates_from_map(s_map)
        r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
        s_map.data[r > 1] = self.fill_value
        return s_map


class FeaturePatchEditor(Editor):

    def __init__(self, patch_shape=(512, 512)):
        self.patch_shape = patch_shape

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")  # ignore warnings
        hpc_coords = all_coordinates_from_map(s_map)
        r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs

        data = np.copy(s_map.data)
        # randomly select limb
        if random.random() < 0.2:
            data[r > 0.9] = np.nan
            data[(r > 0.9) & (r < 1.)] = -1
        else:
            data[r > 0.9] = np.nan

        pixel_pos = np.argwhere(data == np.nanmin(data))
        pixel_pos = pixel_pos[randint(0, len(pixel_pos) - 1)]
        pixel_pos = np.min([pixel_pos[0], data.shape[0] - self.patch_shape[0] // 2]), \
                    np.min([pixel_pos[1], data.shape[1] - self.patch_shape[1] // 2])
        pixel_pos = np.max([pixel_pos[0], self.patch_shape[0] // 2]), \
                    np.max([pixel_pos[1], self.patch_shape[1] // 2])

        pixel_pos = np.array(pixel_pos) * u.pix
        center = s_map.pixel_to_world(pixel_pos[1], pixel_pos[0])

        arcs_frame = s_map.scale[0] * (self.patch_shape[0] / 2 * u.pix)
        s_map = s_map.submap(SkyCoord([center.Tx - arcs_frame, center.Tx + arcs_frame],
                                      [center.Ty - arcs_frame, center.Ty + arcs_frame],
                                      frame=s_map.coordinate_frame))

        return s_map


class RandomPatchEditor(Editor):
    def __init__(self, patch_shape):
        self.patch_shape = patch_shape

    def call(self, data, **kwargs):
        assert data.shape[1] >= self.patch_shape[0], 'Invalid data shape: %s' % str(data.shape)
        assert data.shape[2] >= self.patch_shape[1], 'Invalid data shape: %s' % str(data.shape)
        x = randint(0, data.shape[1] - self.patch_shape[0])
        y = randint(0, data.shape[2] - self.patch_shape[1])
        patch = data[:, x:x + self.patch_shape[0], y:y + self.patch_shape[1]]
        assert np.std(patch) != 0, 'Invalid patch found (all values %f)' % np.mean(patch)
        return patch

class SliceEditor(Editor):

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def call(self, data, **kwargs):
        return data[self.start:self.stop]

class BrightestPixelPatchEditor(Editor):
    def __init__(self, patch_shape, idx=0):
        self.patch_shape = patch_shape
        self.idx = idx

    def call(self, data, **kwargs):
        assert data.shape[1] >= self.patch_shape[0], 'Invalid data shape: %s' % str(data.shape)
        assert data.shape[2] >= self.patch_shape[1], 'Invalid data shape: %s' % str(data.shape)

        smoothed = ndimage.gaussian_filter(data[self.idx], sigma=5)
        pixel_pos = np.argwhere(smoothed == np.nanmax(smoothed))
        pixel_pos = pixel_pos[randint(0, len(pixel_pos) - 1)]
        pixel_pos = np.min([pixel_pos[0], smoothed.shape[0] - self.patch_shape[0] // 2]), np.min(
            [pixel_pos[1], smoothed.shape[1] - self.patch_shape[1] // 2])
        pixel_pos = np.max([pixel_pos[0], self.patch_shape[0] // 2]), np.max([pixel_pos[1], self.patch_shape[1] // 2])

        x = pixel_pos[0]
        y = pixel_pos[1]
        patch = data[:,
                x - int(np.floor(self.patch_shape[0] / 2)):x + int(np.ceil(self.patch_shape[0] / 2)),
                y - int(np.floor(self.patch_shape[1] / 2)):y + int(np.ceil(self.patch_shape[1] / 2)), ]
        assert np.std(patch) != 0, 'Invalid patch found (all values %f)' % np.mean(patch)
        return patch


class EITCheckEditor(Editor):

    def call(self, s_map, **kwargs):
        assert np.all(np.logical_not(np.isnan(s_map.data))), 'Found missing block %s' % s_map.date.datetime.isoformat()
        assert 'N_MISSING_BLOCKS =    0' in s_map.meta['comment'], 'Found missing block %s: %s' % (
            s_map.date.datetime.isoformat(), s_map.meta['comment'])
        return s_map


class PaddingEditor(Editor):
    def __init__(self, patch_shape):
        self.patch_shape = patch_shape

    def call(self, data, **kwargs):
        s = data.shape
        p = self.patch_shape
        x_pad = (p[0] - s[-2]) / 2
        y_pad = (p[1] - s[-1]) / 2
        pad = [(int(np.floor(x_pad)), int(np.ceil(x_pad))),
               (int(np.floor(y_pad)), int(np.ceil(y_pad)))]
        if len(s) == 3:
            pad.insert(0, (0, 0))
        return np.pad(data, pad, 'constant', constant_values=np.min(data))

class ReductionEditor(Editor):

    def call(self, data, **kwargs):
        s = data.shape
        p = kwargs['patch_shape']
        x_pad = (s[-2] - p[0]) / 2
        y_pad = (s[-1] - p[1]) / 2
        pad = [(int(np.floor(x_pad)), int(np.ceil(x_pad))),
               (int(np.floor(y_pad)), int(np.ceil(y_pad)))]
        return data[..., pad[0][0]:-pad[0][1], pad[1][0]:-pad[1][1]]


class PassEditor(Editor):

    def call(self, data, **kwargs):
        return data

class LambdaEditor(Editor):

    def __init__(self, f):
        self.f = f

    def call(self, data, **kwargs):
        return self.f(data, **kwargs)


class LimbDarkeningCorrectionEditor(Editor):

    def __init__(self, limb_offset=0.99):
        self.limb_offset = limb_offset

    def call(self, s_map, **kwargs):
        coords = all_coordinates_from_map(s_map)
        radial_distance = (np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / s_map.rsun_obs).value
        radial_distance[radial_distance >= self.limb_offset] = np.NaN
        ideal_correction = np.cos(radial_distance * np.pi / 2)

        condition = np.logical_not(np.isnan(np.ravel(ideal_correction)))
        map_list = np.ravel(s_map.data)[condition]
        correction_list = np.ravel(ideal_correction)[condition]

        fit = np.polyfit(correction_list, map_list, 5)
        poly_fit = np.poly1d(fit)

        map_correction = poly_fit(ideal_correction)
        corrected_map = s_map.data / map_correction

        return Map(corrected_map.astype(np.float32), s_map.meta)


def get_local_correction_table():
    path = os.path.join(Path.home(), 'aiapy', 'correction_table.dat')
    if os.path.exists(path):
        return get_correction_table(path)
    os.makedirs(os.path.join(Path.home(), 'aiapy'), exist_ok=True)
    correction_table = get_correction_table()
    astropy.io.ascii.write(correction_table, path)
    return correction_table
