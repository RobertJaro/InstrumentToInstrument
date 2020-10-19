import gc
import glob
import logging
import os
import random
from enum import Enum
from typing import List

import numpy as np
from astropy.visualization import ImageNormalize, LinearStretch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from iti.data.editor import Editor, LoadMapEditor, KSOPrepEditor, NormalizeRadiusEditor, \
    MapToDataEditor, ImageNormalizeEditor, ReshapeEditor, sdo_norms, NormalizeEditor, \
    AIAPrepEditor, RemoveOffLimbEditor, StackEditor, soho_norms, NanEditor, LoadFITSEditor, \
    KSOFilmPrepEditor, ScaleEditor, ExpandDimsEditor, FeaturePatchEditor, EITCheckEditor, NormalizeExposureEditor, \
    PassEditor, BrightestPixelPatchEditor, secchi_norms, LimbDarkeningCorrectionEditor, ContrastNormalizeEditor


class Norm(Enum):
    CONTRAST = 'contrast'
    IMAGE = 'image'
    PEAK = 'adjusted'
    NONE = 'none'


class BaseDataset(Dataset):

    def __init__(self, data, editors: List[Editor]):
        self.data = data
        self.editors = editors

        logging.info("Using {} samples".format(len(self.data)))
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.convertData(self.data[idx])

    def sample(self, n_samples):
        it = DataLoader(self, batch_size=1, shuffle=True, num_workers=4).__iter__()
        samples = []
        while len(samples) < n_samples:
            try:
                samples.append(next(it).detach().numpy()[0])
            except Exception as ex:
                logging.error(str(ex))
                continue
        del it
        return np.array(samples)

    def convertData(self, data):
        kwargs = {}
        for editor in self.editors:
            data, kwargs = editor.convert(data, **kwargs)
        return data

    def addEditor(self, editor):
        self.editors.append(editor)


class StorageDataset(Dataset):
    def __init__(self, dataset, store_dir, ext_editors=[]):
        self.dataset = dataset
        self.store_dir = store_dir
        self.ext_editors = ext_editors
        os.makedirs(store_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        store_path = os.path.join(self.store_dir, '%d.npy' % idx)
        if os.path.exists(store_path):
            data = np.load(store_path)
            data = self.convertData(data)
            return data
        data = self.dataset[idx]
        np.save(store_path, data)
        data = self.convertData(data)
        return data

    def convertData(self, data):
        kwargs = {}
        for editor in self.ext_editors:
            data, kwargs = editor.convert(data, **kwargs)
        return data

    def sample(self, n_samples):
        it = DataLoader(self, batch_size=1, shuffle=True, num_workers=4).__iter__()
        samples = []
        while len(samples) < n_samples:
            try:
                samples.append(next(it).detach().numpy()[0])
            except Exception as ex:
                logging.error(str(ex))
                continue
        del it
        return np.array(samples)

    def convert(self, n_worker):
        it = DataLoader(self, batch_size=1, shuffle=False, num_workers=n_worker).__iter__()
        for _ in tqdm(range(len(self.dataset))):
            try:
                next(it)
                gc.collect()
            except StopIteration:
                return
            except Exception as ex:
                logging.error(str(ex))
                continue


class KSODataset(BaseDataset):

    def __init__(self, path, resolution=256, ext="*.fts.gz", limit=None):
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))
        if limit:
            map_paths = random.sample(map_paths, limit)

        editors = [LoadMapEditor(),
                   KSOPrepEditor(),
                   NormalizeRadiusEditor(resolution),
                   MapToDataEditor(),
                   ImageNormalizeEditor(0, 1000),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)

class KSOFlatDataset(BaseDataset):

    def __init__(self, path, resolution=256, ext="*.fts.gz", limit=None):
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))
        if limit:
            map_paths = random.sample(map_paths, limit)

        editors = [LoadMapEditor(),
                   KSOPrepEditor(),
                   NormalizeRadiusEditor(resolution, 0),
                   LimbDarkeningCorrectionEditor(),
                   MapToDataEditor(),
                   ContrastNormalizeEditor(),
                   ImageNormalizeEditor(-5, 5),
                   NanEditor(-1),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)


class KSOFilmDataset(BaseDataset):

    def __init__(self, path, resolution=256, ext="*.fts.gz", limit=None):
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))
        if limit:
            map_paths = random.sample(map_paths, limit)

        editors = [LoadFITSEditor(),
                   KSOFilmPrepEditor(),
                   NormalizeRadiusEditor(resolution, 0),
                   LimbDarkeningCorrectionEditor(),
                   MapToDataEditor(),
                   ContrastNormalizeEditor(),
                   ImageNormalizeEditor(-5, 5),
                   NanEditor(-1),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)


class SDODataset(BaseDataset):

    def __init__(self, path, patch_shape=None, **kwargs):
        data_sets = [AIADataset(os.path.join(path, '171'), 171, **kwargs),
                     AIADataset(os.path.join(path, '193'), 193, **kwargs),
                     AIADataset(os.path.join(path, '211'), 211, **kwargs),
                     AIADataset(os.path.join(path, '304'), 304, **kwargs),
                     HMIDataset(os.path.join(path, '6173'), 'mag', **kwargs)
                     ]
        # align data in time
        basenames = [[os.path.basename(path) for path in data_set.data] for data_set in data_sets]
        basenames = set(basenames[0]).intersection(*basenames)
        for data_set in data_sets:
            data_set.data = sorted([path for path in data_set.data if os.path.basename(path) in basenames])

        editors = [StackEditor(data_sets)]
        if patch_shape is not None:
            editors.append(BrightestPixelPatchEditor(patch_shape))
        super().__init__(range(len(data_sets[0])), editors)


class SOHODataset(BaseDataset):

    def __init__(self, path, patch_shape=None, **kwargs):
        data_sets = [EITDataset(os.path.join(path, 'eit_171'), 171, **kwargs),
                     EITDataset(os.path.join(path, 'eit_195'), 195, **kwargs),
                     EITDataset(os.path.join(path, 'eit_284'), 284, **kwargs),
                     EITDataset(os.path.join(path, 'eit_304'), 304, **kwargs),
                     MDIDataset(os.path.join(path, 'mdi_mag'), **kwargs)
                     ]
        # align data in time
        basenames = [[os.path.basename(path) for path in data_set.data] for data_set in data_sets]
        basenames = set(basenames[0]).intersection(*basenames)
        for data_set in data_sets:
            data_set.data = sorted([path for path in data_set.data if os.path.basename(path) in basenames])

        editors = [StackEditor(data_sets)]
        if patch_shape is not None:
            editors.append(BrightestPixelPatchEditor(patch_shape))
        super().__init__(range(len(data_sets[0])), editors)


class STEREODataset(BaseDataset):

    def __init__(self, path, patch_shape=None, **kwargs):
        data_sets = [SECCHIDataset(os.path.join(path, 'secchi_171'), 171, **kwargs),
                     SECCHIDataset(os.path.join(path, 'secchi_195'), 195, **kwargs),
                     SECCHIDataset(os.path.join(path, 'secchi_284'), 284, **kwargs),
                     SECCHIDataset(os.path.join(path, 'secchi_304'), 304, **kwargs),
                     ]
        # align data in time
        basenames = [[os.path.basename(path) for path in data_set.data] for data_set in data_sets]
        basenames = set(basenames[0]).intersection(*basenames)
        for data_set in data_sets:
            data_set.data = sorted([path for path in data_set.data if os.path.basename(path) in basenames])

        editors = [StackEditor(data_sets)]
        if patch_shape is not None:
            editors.append(BrightestPixelPatchEditor(patch_shape))
        super().__init__(range(len(data_sets[0])), editors)


class STEREOMagnetogramDataset(BaseDataset):

    def __init__(self, path, patch_shape=None, **kwargs):
        ds_171 = SECCHIDataset(os.path.join(path, 'secchi_171'), 171, **kwargs)
        ds_195 = SECCHIDataset(os.path.join(path, 'secchi_195'), 195, **kwargs)
        ds_284 = SECCHIDataset(os.path.join(path, 'secchi_284'), 284, **kwargs)
        ds_304 = SECCHIDataset(os.path.join(path, 'secchi_304'), 304, **kwargs)
        ds_zeros = ZerosDataset(len(ds_171), **kwargs)
        data_sets = [ds_171, ds_195, ds_284, ds_304, ds_zeros, ]
        editors = [StackEditor(data_sets)]
        if patch_shape is not None:
            editors.append(BrightestPixelPatchEditor(patch_shape))
        super().__init__(range(len(data_sets[0])), editors)


class EITDataset(BaseDataset):

    def __init__(self, path, wavelength, resolution=1024, ext='*.fits'):
        norm = soho_norms[wavelength]
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))

        editors = [LoadMapEditor(),
                   EITCheckEditor(),
                   NormalizeRadiusEditor(resolution),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)


class MDIDataset(BaseDataset):

    def __init__(self, path, resolution=1024, ext='*.fits'):
        norm = soho_norms[6173]
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution),
                   RemoveOffLimbEditor(),
                   MapToDataEditor(),
                   NanEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)


class AIADataset(BaseDataset):

    def __init__(self, path, wavelength, resolution=2048, ext='*.fits'):
        norm = sdo_norms[wavelength]
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution),
                   AIAPrepEditor(),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)


class HMIDataset(BaseDataset):

    def __init__(self, path, id, resolution=2048, ext='*.fits'):
        norm = sdo_norms[id]
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution),
                   RemoveOffLimbEditor(),
                   MapToDataEditor(),
                   NanEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)


class HMIContinuumDataset(BaseDataset):

    def __init__(self, path, patch_shape=None, ext='*.fits', **kwargs):
        norm = sdo_norms['continuum']
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))

        editors = [LoadMapEditor(),
                   ScaleEditor(0.6),
                   FeaturePatchEditor(patch_shape) if patch_shape is not None else PassEditor(),
                   MapToDataEditor(),
                   NanEditor(),
                   NormalizeEditor(norm),
                   ExpandDimsEditor()]
        super().__init__(map_paths, editors=editors, **kwargs)


class HinodeDataset(BaseDataset):

    def __init__(self, path, ext='*.fits'):
        norm = ImageNormalize(vmin=0, vmax=50000, stretch=LinearStretch(), clip=True)
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True)) if isinstance(path, str) else path

        editors = [LoadMapEditor(),
                   ScaleEditor(0.15),
                   NormalizeExposureEditor(),
                   MapToDataEditor(),
                   NanEditor(),
                   NormalizeEditor(norm),
                   ExpandDimsEditor()]
        super().__init__(map_paths, editors=editors)


class SECCHIDataset(BaseDataset):

    def __init__(self, path, wavelength, resolution=1024, ext='*.fits'):
        norm = secchi_norms[wavelength]
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(1024),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)


class ZerosDataset(Dataset):

    def __init__(self, length, resolution=1024, **kwargs):
        self.shape = (1, resolution, resolution)
        self.length = length

        super().__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return np.zeros(self.shape)
