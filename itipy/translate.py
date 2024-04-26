import os
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Tuple
from urllib import request

import astropy.units as u
import numpy as np
import torch
from skimage.util import view_as_blocks
from sunpy.map import Map, make_fitswcs_header, all_coordinates_from_map

from itipy.data.dataset import SOHODataset, HMIContinuumDataset, STEREODataset, KSOFlatDataset, KSOFilmDataset
from itipy.data.editor import PaddingEditor, sdo_norms, hinode_norms, UnpaddingEditor


class InstrumentToInstrument:

    def __init__(self, model_name, model_path=None, device=None, depth_generator=3, patch_factor=0, n_workers=4):
        self.patch_factor = patch_factor
        self.depth_generator = depth_generator
        # Load Model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path is None:
            model_path = self._getModelPath(model_name)
        self.generator = torch.load(model_path, map_location=device)
        self.generator.to(device)
        self.generator.eval()
        self.device = device
        self.n_workers = n_workers

    def translate(self, *args, **kwargs):
        raise NotImplementedError()

    def _translateDataset(self, dataset):
        with Pool(self.n_workers) as pool:
            for img, kwargs in pool.imap(dataset.convertData, dataset.data):
                #
                original_shape = img.shape
                img = np.array(img.data)  # remove np mask information
                #
                min_dim = min(
                    [i for i in range(img.shape[1], img.shape[1] * 2 ** (self.depth_generator + self.patch_factor))
                     if i % 2 ** (self.depth_generator + self.patch_factor) == 0])  # find min dim
                target_shape = (min_dim, min_dim)
                padding_editor = PaddingEditor(target_shape)
                # pad
                padded_img = padding_editor.call(img)
                padded_img = np.nan_to_num(padded_img, nan=np.nanmin(padded_img))
                # translate
                with torch.no_grad():
                    if self.patch_factor > 0:
                        iti_img = self._translateBlocks(padded_img, self.patch_factor)
                    else:
                        iti_img = self.generator(torch.tensor(padded_img).float().to(self.device).unsqueeze(0))
                        iti_img = iti_img[0].detach().cpu().numpy()
                # unpad
                scaling = iti_img.shape[-1] / padded_img.shape[-1]
                iti_img = UnpaddingEditor([p * scaling for p in original_shape[1:]]).call(iti_img)
                #
                ref_meta = [k['header'] for k in kwargs['kwargs_list']] if 'kwargs_list' in kwargs else [
                    kwargs['header']]
                # use last meta data as reference for additional observables
                ref_meta += [ref_meta[-1]] * (len(iti_img) - len(ref_meta))
                #
                ref_img = img.tolist()
                ref_img += [ref_img[-1]] * (len(iti_img) - len(ref_img))  # extend list
                ref_img = np.array(ref_img)
                #
                # create meta for additional channels
                maps = [Map(d, self._createMeta(d, ref_d, meta)) for d, ref_d, meta in zip(iti_img, ref_img, ref_meta)]
                maps = maps[0] if len(maps) == 1 else maps
                yield maps, img, iti_img

    def _createMeta(self, data, ref_data, ref_meta):
        scaling = data.shape[0] / ref_data.shape[0]
        ref_map = Map(ref_data, ref_meta)  # copy observer
        scale = (ref_meta['cdelt1'] / scaling, ref_meta['cdelt2'] / scaling)
        coord = ref_map.reference_coordinate
        wl = ref_map.wavelength if ref_map.waveunit is not None else None
        meta = make_fitswcs_header(data, coord, rotation_matrix=ref_map.rotation_matrix, scale=scale * u.arcsec / u.pix,
                                   observatory='ITI', instrument='ITI - ' + ref_map.instrument, wavelength=wl,
                                   exposure=1 * u.s, )
        return meta

    def _translateBlocks(self, img, n_patches):
        patch_dim = img.shape[-1] // n_patches
        #
        patch_shape = (img.shape[0], patch_dim, patch_dim)
        patches = view_as_blocks(img, patch_shape)
        patches = np.reshape(patches, (-1, *patch_shape))
        iti_patches = []
        with torch.no_grad():
            for patch in patches:
                iti_patch = self.generator(torch.tensor(patch).float().to(self.device).unsqueeze(0))
                iti_patches.append(iti_patch[0].detach().cpu().numpy())
        #
        iti_patches = np.array(iti_patches)
        iti_patches = iti_patches.reshape((n_patches, n_patches,
                                           iti_patches.shape[1], iti_patches.shape[2], iti_patches.shape[3]))
        iti_img = np.moveaxis(iti_patches, [0, 1], [1, 3]).reshape((iti_patches.shape[2],
                                                                    iti_patches.shape[0] * iti_patches.shape[3],
                                                                    iti_patches.shape[1] * iti_patches.shape[4]))
        #
        return iti_img

    def _getModelPath(self, model_name):
        model_path = os.path.join(Path.home(), '.iti', model_name)
        os.makedirs(os.path.join(Path.home(), '.iti'), exist_ok=True)
        if not os.path.exists(model_path):
            request.urlretrieve('http://kanzelhohe.uni-graz.at/iti/' + model_name, filename=model_path)
        return model_path

    def _adjustMeta(self, meta, new_data, scale_factor):
        # Update image scale and number of pixels
        new_meta = meta.copy()

        # Update metadata
        new_meta['cdelt1'] /= scale_factor
        new_meta['cdelt2'] /= scale_factor
        if 'CD1_1' in new_meta:
            new_meta['CD1_1'] /= scale_factor
            new_meta['CD2_1'] /= scale_factor
            new_meta['CD1_2'] /= scale_factor
            new_meta['CD2_2'] /= scale_factor
        new_meta['crpix1'] = (new_data.shape[1] + 1) / 2.
        new_meta['crpix2'] = (new_data.shape[0] + 1) / 2.
        s_map = Map(new_data, new_meta)
        lon, lat = s_map._get_lon_lat(s_map.center.frame)
        new_meta['crval1'] = lon.value
        new_meta['crval2'] = lat.value
        new_meta['naxis1'] = new_data.shape[1]
        new_meta['naxis2'] = new_data.shape[0]
        history = new_meta.get('history') + '; ' if 'history' in new_meta else ''
        new_meta['history'] = history + 'ITI enhanced'
        return new_meta


class InstrumentConverter:

    def _convertDataset(self, *datasets, n_workers=4) -> Tuple[np.ndarray, List]:
        images = []
        metas = []
        with Pool(n_workers) as pool:
            for data_sample in zip(*[pool.imap(ds.convertData, ds.data) for ds in datasets]):
                #
                images += [d for d, kwargs in data_sample]
                metas += [kwargs for d, kwargs in data_sample]
        return np.concatenate(images), metas


class SOHOToSDO(InstrumentToInstrument):

    def __init__(self, model_name='soho_to_sdo_v0_2.pt', **kwargs):
        super().__init__(model_name, **kwargs)
        self.norms = [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304], sdo_norms['mag']]

    def translate(self, path, basenames=None, **kwargs):
        soho_dataset = SOHODataset(path, basenames=basenames, **kwargs)
        for maps, img, iti_img in self._translateDataset(soho_dataset):
            yield [Map(norm.inverse((s_map.data + 1) / 2), self.toSDOMeta(s_map.meta, instr))
                   for s_map, norm, instr in zip(maps, self.norms, ['AIA'] * 4 + ['HMI'])]

    def toSDOMeta(self, meta, instrument):
        wl_map = {171: 171, 195: 193, 284: 211, 304: 304, 6768: 6173, 0: 0}
        new_meta = meta.copy()
        new_meta['obsrvtry'] = 'SOHO-to-SDO'
        new_meta['telescop'] = 'sdo'
        new_meta['instrume'] = instrument
        new_meta['WAVELNTH'] = wl_map[meta.get('WAVELNTH', 0)]
        new_meta['waveunit'] = 'angstrom'
        return new_meta


class SOHOToSDOEUV(SOHOToSDO):

    def __init__(self, model_name='soho_to_sdo_euv_v0_1.pt', **kwargs):
        super().__init__(model_name, **kwargs)
        self.norms = [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304]]

    def translate(self, path, basenames=None):
        soho_dataset = SOHODataset(path, basenames=basenames, wavelengths=[171, 195, 284, 304])
        for maps, img, iti_img in self._translateDataset(soho_dataset):
            yield [Map(norm.inverse((s_map.data + 1) / 2), self.toSDOMeta(s_map.meta, instr))
                   for s_map, norm, instr in zip(maps, self.norms, ['AIA'] * 4)]


class STEREOToSDO(InstrumentToInstrument):

    def __init__(self, model_name='stereo_to_sdo_v0_2.pt', **kwargs):
        super().__init__(model_name, **kwargs)

    def translate(self, path, basenames=None, return_arrays=False):
        soho_dataset = STEREODataset(path, basenames=basenames)
        for result, inputs, outputs in self._translateDataset(soho_dataset):
            norms = [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304]]
            result = [Map(norm.inverse((s_map.data + 1) / 2), self.toSDOMeta(s_map.meta, instrument, wl))
                      for s_map, norm, instrument, wl in
                      zip(result, norms, ['AIA'] * 4, [171, 193, 211, 304])]
            if return_arrays:
                yield result, inputs, outputs
            else:
                yield result

    def toSDOMeta(self, meta, instrument, wl):
        new_meta = meta.copy()
        new_meta['obsrvtry'] = 'SOHO-to-SDO'
        new_meta['telescop'] = 'sdo'
        new_meta['instrume'] = instrument
        new_meta['WAVELNTH'] = wl
        new_meta['waveunit'] = 'angstrom'
        return new_meta


class STEREOToSDOMagnetogram(InstrumentToInstrument):

    def __init__(self, model_name='stereo_to_sdo_mag_v0_2.pt', **kwargs):
        super().__init__(model_name, **kwargs)

    def translate(self, path, basenames=None, return_arrays=False):
        soho_dataset = STEREODataset(path, basenames=basenames)
        for result, inputs, outputs in self._translateDataset(soho_dataset):
            norms = [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304]]
            result = [Map(norm.inverse((s_map.data + 1) / 2), self.toSDOMeta(s_map.meta, 'AIA', wl))
                      for s_map, norm, wl in zip(result[:-1], norms, [171, 193, 211, 304])] + \
                     [self._createMagnetogramMap(result[-1].data, result[-1].meta)]
            if return_arrays:
                yield result, inputs, outputs
            else:
                yield result

    def _createMagnetogramMap(self, data, meta):
        v_max = sdo_norms['mag'].vmax
        s_map = Map((data + 1) / 2 * v_max, self.toSDOMeta(meta, 'HMI', 6173))
        hpc_coords = all_coordinates_from_map(s_map)
        r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
        s_map.data[r > 1] = np.nan
        return s_map

    def toSDOMeta(self, meta, instrument, wl):
        new_meta = meta.copy()
        new_meta['obsrvtry'] = 'SOHO-to-SDO'
        new_meta['telescop'] = 'sdo'
        new_meta['instrume'] = instrument
        new_meta['WAVELNTH'] = wl
        new_meta['waveunit'] = 'angstrom'
        return new_meta


class KSOLowToHigh(InstrumentToInstrument):
    def __init__(self, model_name='kso_low_to_high_v0_2.pt', resolution=512, **kwargs):
        super().__init__(model_name, **kwargs)
        self.resolution = resolution

    def translate(self, paths, return_arrays=False, **kwargs):
        ds = KSOFlatDataset(paths, self.resolution, **kwargs)
        for result, inputs, outputs in self._translateDataset(ds):
            if return_arrays:
                yield result, inputs, outputs
            else:
                yield result


class KSOFilmToCCD(InstrumentToInstrument):
    def __init__(self, model_name='kso_film_to_ccd_v0_1.pt', resolution=512, **kwargs):
        super().__init__(model_name, **kwargs)
        self.resolution = resolution

    def translate(self, paths, return_arrays=False):
        ds = KSOFilmDataset(paths, self.resolution)
        for result, inputs, outputs in self._translateDataset(ds):
            if return_arrays:
                yield result, inputs, outputs
            else:
                yield result


class HMIToHinode(InstrumentToInstrument):
    def __init__(self, model_name='hmi_to_hinode_v0_2.pt', **kwargs):
        super().__init__(model_name, **kwargs)

    def translate(self, paths):
        ds = HMIContinuumDataset(paths)
        for s_map, input, output in self._translateDataset(ds):
            norm = hinode_norms['continuum']
            s_map = Map(norm.inverse((s_map.data + 1) / 2), s_map.meta)
            yield s_map


class KSOFlatConverter(InstrumentConverter):

    def __init__(self, resolution, **kwargs):
        super().__init__(**kwargs)
        self.resolution = resolution

    def convert(self, paths):
        ds = KSOFlatDataset(paths, self.resolution)
        return self._convertDataset(ds)
