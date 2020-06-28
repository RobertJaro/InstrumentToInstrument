import os
import shutil
import warnings
from abc import ABC, abstractmethod

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch
from skimage.transform import pyramid_reduce
from sunpy.map import Map


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


class ContrastNormalizeEditor(Editor):

    def __init__(self, use_median=False, threshold=False):
        self.use_median = use_median
        self.threshold = threshold

    def call(self, data, **kwargs):
        shift = np.median(data, (0, 1), keepdims=True) if self.use_median else np.mean(data, (0, 1), keepdims=True)
        data = (data - shift) / (np.std(data, (0, 1), keepdims=True) + 10e-8)
        if self.threshold:
            data[data > self.threshold] = self.threshold
            data[data < -self.threshold] = -self.threshold
            data /= self.threshold
        return data


class ImageNormalizeEditor(Editor):

    def __init__(self, vmin=0, vmax=1000, stretch=LinearStretch()):
        self.norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch, clip=True)

    def call(self, data, **kwargs):
        data = self.norm(data).data * 2 - 1
        return data


class ReshapeEditor(Editor):

    def __init__(self, shape):
        self.shape = shape

    def call(self, data, **kwargs):
        data = data[:self.shape[1], :self.shape[2]]
        return np.reshape(data, self.shape)


class NanEditor(Editor):
    def call(self, data, **kwargs):
        data = np.nan_to_num(data)
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


class NormalizeRadiusEditor(Editor):
    def __init__(self, arcs_pp, resolution):
        self.arcs_pp = arcs_pp
        self.resolution = resolution
        super(NormalizeRadiusEditor, self).__init__()

    def call(self, s_map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            target_scale = self.arcs_pp / (959.21 / s_map.rsun_obs.value) * (
                        s_map.data.shape[0] / self.resolution)  # normalize solar radius
            scale_factor = s_map.scale[0].value / target_scale
            s_map = s_map.rotate(recenter=True, scale=scale_factor, missing=s_map.min())
            arcs_frame = (self.resolution / 2) * s_map.scale[0].value * u.arcsec
            s_map = s_map.submap(SkyCoord([-arcs_frame, arcs_frame] * u.arcsec,
                                          [-arcs_frame, arcs_frame] * u.arcsec,
                                          frame=s_map.coordinate_frame))
            s_map.meta['r_sun'] = s_map.rsun_obs.value / s_map.meta['cdelt1']

            return s_map


class PyramidRescaleEditor(Editor):

    def __init__(self, scale=2):
        self.scale = scale

    def call(self, data, **kwargs):
        data = pyramid_reduce(data, downscale=self.scale)
        return data


class LoadNumpyEditor(Editor):

    def call(self, data, **kwargs):
        return np.load(data)
