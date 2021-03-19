import glob
import gzip
import os
import shutil
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Union, List
from urllib import request

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.util import view_as_blocks
from sunpy.map import Map

from iti.data.dataset import SOHODataset, KSODataset, HMIContinuumDataset, STEREODataset
from iti.data.editor import PaddingEditor, sdo_norms, hinode_norms, UnpaddingEditor


class InstrumentToInstrument:

    def __init__(self, model_name, model_path=None, device=None, depth_generator=3, patch_factor=0):
        self.patch_factor = patch_factor
        self.depth_generator = depth_generator
        # Load Model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path is None:
            model_path = self._getModelPath(model_name)
        self.generator = torch.load(model_path)
        self.generator.to(device)
        self.generator.eval()
        self.device = device

    def translate(self, *args, **kwargs):
        raise NotImplementedError()

    def _translateDataset(self, *datasets, n_workers=4) -> List[Union[List[Map], Map]]:
        result = []
        with Pool(n_workers) as pool:
            for data_sample in zip(*[pool.imap(ds.convertData, ds.data) for ds in datasets]):
                #
                img = np.concatenate([d for d, kwargs in data_sample])
                original_shape = img.shape
                #
                min_dim = min([i for i in range(img.shape[1], img.shape[1] * 2 ** (self.depth_generator + self.patch_factor))
                           if i % 2 ** (self.depth_generator + self.patch_factor) == 0])  # find min dim
                target_shape = (min_dim, min_dim)
                padding_editor = PaddingEditor(target_shape)
                #
                img = padding_editor.call(img.data)
                with torch.no_grad():
                    if self.patch_factor > 0:
                        iti_img = self._translateBlocks(img, self.patch_factor)
                    else:
                        iti_img = self.generator(torch.tensor(img).float().to(self.device).unsqueeze(0))
                        iti_img = iti_img[0].detach().cpu().numpy()
                scaling = iti_img.shape[-1] / img.shape[-1]
                iti_img = UnpaddingEditor([p * scaling for p in original_shape[1:]]).call(iti_img)
                #
                metas = [self._adjustMeta(kwargs['header'], iti_img[0], scaling) for _, kwargs in data_sample]
                #
                maps = [Map(d, meta) for d, meta in zip(iti_img, metas)]
                maps = maps[0] if len(maps) == 1 else maps
                result.append(maps)
        return result

    def _translateBlocks(self, img, n_patches):
        patch_dim = img.shape[-1] // n_patches
        #
        patch_shape = (img.shape[0], patch_dim, patch_dim)
        patches = view_as_blocks(img, patch_shape)
        patches = np.reshape(patches, (-1, *patch_shape))
        iti_patches = []
        with torch.no_grad():
            for patch in patches:
                iti_patch = self.generator(torch.tensor(patch).float().cuda().unsqueeze(0))
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
        model_path = os.path.join(Path.home(), 'iti', model_name)
        if not os.path.exists(model_path):
            request.urlretrieve('http://kanzelhohe.uni-graz.at/solarnet-campaign/iti/' + model_name,
                                filename=model_path)
        return model_path

    def _adjustMeta(self, meta, new_data, scale_factor):
        # Update image scale and number of pixels
        new_meta = meta.copy()

        # Update metadata
        new_meta['cdelt1'] /= scale_factor
        new_meta['cdelt2'] /= scale_factor
        # if 'CD1_1' in new_meta:
        #     new_meta['CD1_1'] *= scale_factor
        #     new_meta['CD2_1'] *= scale_factor
        #     new_meta['CD1_2'] *= scale_factor
        #     new_meta['CD2_2'] *= scale_factor
        new_meta['crpix1'] = (new_data.shape[0] + 1) / 2.
        new_meta['crpix2'] = (new_data.shape[1] + 1) / 2.
        new_meta['naxis1'] = new_data.shape[1]
        new_meta['naxis2'] = new_data.shape[0]
        history = new_meta.get('history') + '; ' if 'history' in new_meta else ''
        new_meta['history'] = history + 'ITI enhanced'
        return new_meta


class SOHOtoSDO(InstrumentToInstrument):

    def __init__(self, model_name='soho_to_sdo_v0_1.py', **kwargs):
        super().__init__(model_name, **kwargs)

    def translate(self, path, base_names=None):
        soho_dataset = SOHODataset(path, base_names=base_names)
        result = self._translateDataset(*soho_dataset.data_sets)
        norms = [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304], sdo_norms['mag']]
        result = [[Map(norm.inverse((s_map.data + 1) / 2), self.toSDOMeta(s_map.meta))
                   for s_map, norm in zip(maps, norms)]
                  for maps in result]
        return result

    def toSDOMeta(self, meta):
        wl_map = {171: 171, 195: 193, 284: 211, 304: 304, 0: 0}
        new_meta = meta.copy()
        new_meta['obsrvtry'] = 'SOHO-to-SDO'
        new_meta['telescop'] = 'sdo'
        new_meta['instrume'] = 'AIA' if meta['instrume'] == 'EIT' else 'HMI'
        new_meta['WAVELNTH'] = wl_map[meta.get('WAVELNTH', 0)]
        new_meta['waveunit'] = 'angstrom'
        return new_meta

class STEREOtoSDO(InstrumentToInstrument):

    def __init__(self, model_name='stereo_to_sdo_v0_1.py', **kwargs):
        super().__init__(model_name, **kwargs)

    def translate(self, path, base_names=None):
        soho_dataset = STEREODataset(path, base_names=base_names)
        result = self._translateDataset(*soho_dataset.data_sets)
        norms = [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304]]
        result = [[Map(norm.inverse((s_map.data + 1) / 2), self.toSDOMeta(s_map.meta))
                   for s_map, norm in zip(maps, norms)]
                  for maps in result]
        return result

    def toSDOMeta(self, meta):
        wl_map = {171: 171, 195: 193, 284: 211, 304: 304, 0: 0}
        new_meta = meta.copy()
        new_meta['obsrvtry'] = 'SOHO-to-SDO'
        new_meta['telescop'] = 'sdo'
        new_meta['instrume'] = 'AIA' if meta['instrume'] == 'SECCHI' else 'HMI'
        new_meta['WAVELNTH'] = wl_map[meta.get('WAVELNTH', 0)]
        new_meta['waveunit'] = 'angstrom'
        return new_meta

class KSOLowToHigh(InstrumentToInstrument):
    def __init__(self, model_name='kso_low_to_high_v0_1.py', resolution=512, **kwargs):
        super().__init__(model_name, **kwargs)
        self.resolution = resolution

    def translate(self, paths):
        ds = KSODataset(paths, self.resolution)
        result = self._translateDataset(ds)
        return result


class HMIToHinode(InstrumentToInstrument):
    def __init__(self, model_name='hmi_to_hinode_v0_1.pt', **kwargs):
        super().__init__(model_name, **kwargs)

    def translate(self, paths):
        ds = HMIContinuumDataset(paths)
        s_maps = self._translateDataset(ds)
        norm = hinode_norms['continuum']
        result = [Map(norm.inverse((s_map.data + 1) / 2), s_map.meta) for s_map in s_maps]
        return result


if __name__ == '__main__':
    # translator = SOHOtoSDO(model_path='/gss/r.jarolim/iti/soho_sdo_v23/checkpoint_120000.pt')
    # iti_maps = translator.translate('/gss/r.jarolim/data/soho/valid', ['2001-12-01T01:19.fits'])

    base_path = '/gss/r.jarolim/iti/kso_quality_1024_v3'
    prediction_path = os.path.join(base_path, 'translation')
    os.makedirs(prediction_path, exist_ok=True)
    translator = KSOLowToHigh(resolution=1024, model_path=os.path.join(base_path, 'generatorAB.pt'))
    map_files = list(glob.glob('/gss/r.jarolim/data/kso_general/quality2/*.fts.gz'))
    map_files = map_files[::len(map_files) // 100]
    iti_maps = translator.translate(map_files)

    for kso_map, iti_map in zip(Map(map_files), iti_maps):
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        kso_map.meta['WAVEUNIT'] = 'angstrom'
        kso_map.plot(axes=axs[0], title='KSO', vmin=0, vmax=1000)
        iti_map.plot(axes=axs[1], title='ITI', vmin=0, vmax=1000)
        plt.savefig(os.path.join(prediction_path, '%s.jpg') %
                    (iti_map.date.to_datetime().isoformat()))
        #
        file_path = os.path.join(prediction_path, '%s.fits') % iti_map.date.to_datetime().isoformat()
        iti_map.save(file_path, overwrite=True)
        with open(file_path, 'rb') as f_in, gzip.GzipFile(file_path + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)
        os.remove(file_path)
    shutil.make_archive(prediction_path, 'zip', prediction_path)
