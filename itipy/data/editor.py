import os
import os
import random
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from random import randint
from urllib import request

import astropy.io.ascii
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.util import get_correction_table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, AsinhStretch
from dateutil.parser import parse
from scipy import ndimage
from skimage.measure import block_reduce
from skimage.transform import pyramid_reduce
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map, header_helper


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


sdo_norms = {94: ImageNormalize(vmin=0, vmax=340, stretch=AsinhStretch(0.005), clip=True),
             131: ImageNormalize(vmin=0, vmax=1400, stretch=AsinhStretch(0.005), clip=True),
             171: ImageNormalize(vmin=0, vmax=8600, stretch=AsinhStretch(0.005), clip=True),
             193: ImageNormalize(vmin=0, vmax=9800, stretch=AsinhStretch(0.005), clip=True),
             211: ImageNormalize(vmin=0, vmax=5800, stretch=AsinhStretch(0.005), clip=True),
             304: ImageNormalize(vmin=0, vmax=8800, stretch=AsinhStretch(0.001), clip=True),
             335: ImageNormalize(vmin=0, vmax=600, stretch=AsinhStretch(0.005), clip=True),
             1600: ImageNormalize(vmin=0, vmax=4000, stretch=AsinhStretch(0.005), clip=True),
             1700: ImageNormalize(vmin=0, vmax=4000, stretch=AsinhStretch(0.005), clip=True),
             'mag': ImageNormalize(vmin=-3000, vmax=3000, stretch=LinearStretch(), clip=True),
             'continuum': ImageNormalize(vmin=0, vmax=70000, stretch=LinearStretch(), clip=True),
             }

soho_norms = {171: ImageNormalize(vmin=0, vmax=16000, stretch=AsinhStretch(0.005), clip=True),
              195: ImageNormalize(vmin=0, vmax=12000, stretch=AsinhStretch(0.005), clip=True),
              284: ImageNormalize(vmin=0, vmax=2300, stretch=AsinhStretch(0.005), clip=True),
              304: ImageNormalize(vmin=0, vmax=11000, stretch=AsinhStretch(0.005), clip=True),
              6173: ImageNormalize(vmin=-3000, vmax=3000, stretch=LinearStretch(), clip=True),
              }

stereo_norms = {171: ImageNormalize(vmin=0, vmax=6000, stretch=AsinhStretch(0.005), clip=True),
                195: ImageNormalize(vmin=0, vmax=3400, stretch=AsinhStretch(0.005), clip=True),
                284: ImageNormalize(vmin=0, vmax=1300, stretch=AsinhStretch(0.005), clip=True),
                304: ImageNormalize(vmin=0, vmax=18100, stretch=AsinhStretch(0.005), clip=True),
                }

hinode_norms = {'continuum': ImageNormalize(vmin=0, vmax=50000, stretch=LinearStretch(), clip=True),
                'gband': ImageNormalize(vmin=0, vmax=25000, stretch=LinearStretch(), clip=True), }

gregor_norms = {'gband': ImageNormalize(vmin=0, vmax=1.8, stretch=LinearStretch(), clip=True)}


class LoadFITSEditor(Editor):

    def call(self, map_path, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            hdul = fits.open(map_path)
            hdul.verify("fix")
            data, header = hdul[0].data, hdul[0].header
            hdul.close()
        return data, {"header": header}


class LoadMapEditor(Editor):

    def call(self, data, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_map = Map(data)
            s_map.meta['timesys'] = 'tai'  # fix leap seconds
            return s_map, {'path': data}


class LoadGregorGBandEditor(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'wavelnth' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['wavelnth'] == 430.7:
            index = 0
        elif hdul[1].header['wavelnth'] == 430.7:
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        primary_header['cdelt1'] = 70 / 1280
        primary_header['cdelt2'] = 70 / 1280
        #
        g_band = hdul[index::2]
        g_band = sorted(g_band, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in g_band]
        return gregor_maps, {'path': file}


class SubMapEditor(Editor):

    def __init__(self, coords):
        self.coords = coords

    def call(self, s_map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            return s_map.submap(SkyCoord(*self.coords, frame=s_map.coordinate_frame))


class MapToDataEditor(Editor):

    def call(self, s_map, **kwargs):
        return s_map.data, {"header": s_map.meta}


class AddRadialDistanceEditor(Editor):

    def call(self, data, **kwargs):
        s_map = Map(data, kwargs['header'])
        coords = all_coordinates_from_map(s_map)
        radial_distance = (np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / s_map.rsun_obs).value
        radial_distance[radial_distance >= 1] = -1
        return np.stack([s_map.data, radial_distance])


class DataToMapEditor(Editor):

    def call(self, data, **kwargs):
        return Map(data[0], kwargs['header'])


class ContrastNormalizeEditor(Editor):

    def __init__(self, use_median=False, shift=None, normalization=None):
        self.use_median = use_median
        self.shift = shift
        self.normalization = normalization

    def call(self, data, **kwargs):
        if self.shift is None:
            shift = np.nanmedian(data) if self.use_median else np.nanmean(data)
        else:
            shift = self.shift
        if self.normalization is None:
            normalization = np.nanstd(data)
        else:
            normalization = self.normalization
        data = (data - shift) / (normalization + 10e-8)
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
        return np.expand_dims(data, axis=self.axis).astype(np.float32)


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
            kso_map.meta["waveunit"] = "AA"
            kso_map.meta["arcs_pp"] = kso_map.scale[0].value
            if 'exptime' not in kso_map.meta and 'exp_time' in kso_map.meta:
                kso_map.meta['exptime'] = kso_map.meta['exp_time'] / 1000  # ms to s

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
    def __init__(self, calibration='auto'):
        super().__init__()
        assert calibration in ['aiapy', 'auto', 'none',
                               None], "Calibration must be one of: ['aiapy', 'auto', 'none', None]"
        self.calibration = calibration
        self.table = get_auto_calibration_table() if calibration == 'auto' else get_local_correction_table()

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")  # ignore warnings
        if self.calibration == 'auto':
            s_map = self.correct_degradation(s_map, correction_table=self.table)
        elif self.calibration == 'aiapy':
            s_map = correct_degradation(s_map, correction_table=self.table)
        data = np.nan_to_num(s_map.data)
        data = data / s_map.meta["exptime"]
        return Map(data.astype(np.float32), s_map.meta)

    def correct_degradation(self, s_map, correction_table):
        index = correction_table["DATE"].sub(s_map.date.datetime).abs().idxmin()
        num = s_map.meta["wavelnth"]
        return Map(s_map.data / correction_table.iloc[index][f"{int(num):04}"], s_map.meta)


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
    def __init__(self, resolution, padding_factor=0.1, crop=True, fix_irradiance_with_distance=False, **kwargs):
        self.padding_factor = padding_factor
        self.resolution = resolution
        self.crop = crop
        self.fix_irradiance_with_distance = fix_irradiance_with_distance
        super(NormalizeRadiusEditor, self).__init__(**kwargs)

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")  # ignore warnings

        if self.fix_irradiance_with_distance:
            old_meta = s_map.meta.copy()
        r_obs_pix = s_map.rsun_obs / s_map.scale[0]  # Get the solar radius in pixels
        r_obs_pix = (1 + self.padding_factor) * r_obs_pix  # Get the size in pixels of the padded radius 
        scale_factor = self.resolution / (2 * r_obs_pix.value)
        s_map = Map(np.nan_to_num(s_map.data).astype(np.float32), s_map.meta)
        s_map = s_map.rotate(recenter=True, scale=scale_factor, missing=0, order=4)
        if self.crop:
            arcs_frame = (self.resolution / 2) * s_map.scale[0].value
            s_map = s_map.submap(bottom_left=SkyCoord(-arcs_frame * u.arcsec, -arcs_frame * u.arcsec, frame=s_map.coordinate_frame),
                                 top_right=SkyCoord(arcs_frame * u.arcsec, arcs_frame * u.arcsec, frame=s_map.coordinate_frame))
            pad_x = s_map.data.shape[0] - self.resolution
            pad_y = s_map.data.shape[1] - self.resolution
            s_map = s_map.submap(bottom_left=[pad_x // 2, pad_y // 2] * u.pix,
                                 top_right=[pad_x // 2 + self.resolution - 1, pad_y // 2 + self.resolution - 1] * u.pix)
        
        s_map.meta['r_sun'] = s_map.rsun_obs.value / s_map.meta['cdelt1']

        # Virtually move the instrument such that the sun occupies the expected
        # size in the current optics
        if self.fix_irradiance_with_distance:
            # The sun is bigger, not the scaling of the detector
            s_map.meta['rsun_obs'] = old_meta['rsun_obs']*scale_factor
            # This means that cdelt is the same as the old one
            s_map.meta['cdelt1'] = old_meta['cdelt1']
            s_map.meta['cdelt2'] = old_meta['cdelt1']           
            # But we are also closer to the sun
            s_map.meta['dsun_obs'] = (s_map.meta['rsun_ref']/np.tan(s_map.meta['rsun_obs']*u.arcsec)).value
            # Change intensity due to distance change
            s_map.data[:] = s_map.data[:] * (old_meta['dsun_obs']**2)/(s_map.meta['dsun_obs']**2)

        return s_map


class RecenterEditor(Editor):

    def __init__(self, missing=0, order=4, **kwargs):
        self.missing = missing
        self.order = order
        super().__init__(**kwargs)

    def call(self, s_map, **kwargs):
        return s_map.rotate(recenter=True, missing=self.missing, order=self.order)


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


class BlockReduceEditor(Editor):

    def __init__(self, block_size, func=np.mean):
        self.block_size = block_size
        self.func = func

    def call(self, data, **kwargs):
        return block_reduce(data, self.block_size, func=self.func)


class LoadNumpyEditor(Editor):

    def call(self, data, **kwargs):
        return np.load(data)


class StackEditor(Editor):

    def __init__(self, data_sets):
        self.data_sets = data_sets

    def call(self, idx, **kwargs):
        results = [dp.getIndex(idx) for dp in self.data_sets]
        return np.concatenate([img for img, kwargs in results], 0), {'kwargs_list': [kwargs for img, kwargs in results]}


class DistributeEditor(Editor):

    def __init__(self, editors):
        self.editors = editors

    def call(self, data, **kwargs):
        return np.concatenate([self.convertData(d, **kwargs) for d in data], 0)

    def convertData(self, data, **kwargs):
        for editor in self.editors:
            data, kwargs = editor.convert(data, **kwargs)
        return data


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
        patch = np.copy(patch)  # copy from mmep
        assert np.std(patch) != 0, 'Invalid patch found (all values %f)' % np.mean(patch)
        assert not np.any(np.isnan(patch)), 'NaN found'
        return patch


class RandomPatch3DEditor(Editor):
    def __init__(self, patch_shape):
        self.patch_shape = patch_shape

    def call(self, data, **kwargs):
        assert data.shape[0] >= self.patch_shape[0], 'Invalid data shape: %s' % str(data.shape)
        assert data.shape[1] >= self.patch_shape[1], 'Invalid data shape: %s' % str(data.shape)
        assert data.shape[2] >= self.patch_shape[2], 'Invalid data shape: %s' % str(data.shape)
        c = randint(0, data.shape[0] - self.patch_shape[0])
        x = randint(0, data.shape[1] - self.patch_shape[1])
        y = randint(0, data.shape[2] - self.patch_shape[2])
        patch = data[c:c + self.patch_shape[0], x:x + self.patch_shape[1], y:y + self.patch_shape[2]]
        patch = np.copy(patch)  # copy from mmep
        assert np.std(patch) != 0, 'Invalid patch found (all values %f)' % np.mean(patch)
        assert not np.any(np.isnan(patch)), 'NaN found'
        return patch


class SliceEditor(Editor):

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def call(self, data, **kwargs):
        return data[self.start:self.stop]


class BrightestPixelPatchEditor(Editor):
    def __init__(self, patch_shape, idx=0, random_selection=0.2):
        self.patch_shape = patch_shape
        self.idx = idx
        self.random_selection = random_selection

    def call(self, data, **kwargs):
        assert data.shape[1] >= self.patch_shape[0], 'Invalid data shape: %s' % str(data.shape)
        assert data.shape[2] >= self.patch_shape[1], 'Invalid data shape: %s' % str(data.shape)

        if random.random() <= self.random_selection:
            x = randint(0, data.shape[1] - self.patch_shape[0])
            y = randint(0, data.shape[2] - self.patch_shape[1])
            patch = data[:, x:x + self.patch_shape[0], y:y + self.patch_shape[1]]
        else:
            smoothed = ndimage.gaussian_filter(data[self.idx], sigma=5)
            pixel_pos = np.argwhere(smoothed == np.nanmax(smoothed))
            pixel_pos = pixel_pos[randint(0, len(pixel_pos) - 1)]
            pixel_pos = np.min([pixel_pos[0], smoothed.shape[0] - self.patch_shape[0] // 2]), np.min(
                [pixel_pos[1], smoothed.shape[1] - self.patch_shape[1] // 2])
            pixel_pos = np.max([pixel_pos[0], self.patch_shape[0] // 2]), np.max(
                [pixel_pos[1], self.patch_shape[1] // 2])

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

class SOHOFixHeaderEditor(Editor):

    def call(self, s_map, **kwargs):
        s_map.meta['DATE-OBS'] = s_map.meta['DATE_OBS']  # fix date
        s_map.meta['rsun_ref'] = s_map.rsun_meters.value  # preserve solar radius (SOHO fix)
        return s_map

class SECCHIPrepEditor(Editor):

    def __init__(self, degradation=None):
        self.degradation_fit = np.poly1d(degradation) if degradation else False

    def call(self, s_map, **kwargs):
        assert np.all(np.logical_not(np.isnan(s_map.data))), 'Found missing block %s' % s_map.date.datetime.isoformat()
        assert s_map.meta['nmissing'] == 0, 'Found missing block %s: %s' % (
        s_map.date.datetime.isoformat(), s_map.meta['nmissing'])
        assert s_map.meta['NAXIS1'] == 2048 and s_map.meta[
            'NAXIS2'] == 2048, 'Found invalid resolution: %s' % s_map.date.datetime.isoformat()
        if self.degradation_fit:
            x = mdates.date2num(s_map.date.datetime)
            correction = self.degradation_fit(x)
            s_map = Map(s_map.data / correction, s_map.meta)
        return s_map


class PaddingEditor(Editor):
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def call(self, data, **kwargs):
        s = data.shape
        p = self.target_shape
        x_pad = (p[0] - s[-2]) / 2
        y_pad = (p[1] - s[-1]) / 2
        pad = [(int(np.floor(x_pad)), int(np.ceil(x_pad))),
               (int(np.floor(y_pad)), int(np.ceil(y_pad)))]
        if len(s) == 3:
            pad.insert(0, (0, 0))
        return np.pad(data, pad, 'constant', constant_values=np.nan)


class UnpaddingEditor(Editor):
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def call(self, data, **kwargs):
        s = data.shape
        p = self.target_shape
        x_unpad = (s[-2] - p[0]) / 2
        y_unpad = (s[-1] - p[1]) / 2
        #
        unpad = [(None if int(np.floor(y_unpad)) == 0 else int(np.floor(y_unpad)),
                  None if int(np.ceil(y_unpad)) == 0 else -int(np.ceil(y_unpad))),
                 (None if int(np.floor(x_unpad)) == 0 else int(np.floor(x_unpad)),
                  None if int(np.ceil(x_unpad)) == 0 else -int(np.ceil(x_unpad)))]
        data = data[:, unpad[0][0]:unpad[0][1], unpad[1][0]:unpad[1][1]]
        return data


class ReductionEditor(Editor):

    def call(self, data, **kwargs):
        s = data.shape
        p = kwargs['patch_shape']
        x_pad = (s[-2] - p[-2]) / 2
        y_pad = (s[-1] - p[-1]) / 2
        pad = [(int(np.floor(x_pad)), int(np.ceil(x_pad))),
               (int(np.floor(y_pad)), int(np.ceil(y_pad)))]
        if x_pad == 0 and y_pad == 0:
            return data
        if x_pad == 0:
            return data[..., pad[1][0]:-pad[1][1]]
        if y_pad == 0:
            return data[..., pad[0][0]:-pad[0][1], :]
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

        fit = np.polyfit(correction_list, map_list, 4)
        poly_fit = np.poly1d(fit)

        map_correction = poly_fit(ideal_correction)
        corrected_map = s_map.data / map_correction

        return Map(corrected_map, s_map.meta)


def get_local_correction_table():
    path = os.path.join(Path.home(), 'aiapy', 'correction_table.dat')
    if os.path.exists(path):
        return get_correction_table(path)
    os.makedirs(os.path.join(Path.home(), 'aiapy'), exist_ok=True)
    correction_table = get_correction_table()
    astropy.io.ascii.write(correction_table, path)
    return correction_table


def get_auto_calibration_table():
    table_path = os.path.join(Path.home(), '.iti', 'sdo_autocal_table.csv')
    os.makedirs(os.path.join(Path.home(), '.iti'), exist_ok=True)
    if not os.path.exists(table_path):
        request.urlretrieve('http://kanzelhohe.uni-graz.at/iti/sdo_autocal_table.csv', filename=table_path)
    return pd.read_csv(table_path, parse_dates=['DATE'], index_col=0)
