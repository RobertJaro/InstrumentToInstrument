import gc
import glob
import logging
import os
import random
from enum import Enum
from multiprocessing.pool import Pool
from typing import List

import numpy as np
from astropy.visualization import ImageNormalize, LinearStretch
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from iti.data.editor import Editor, LoadMapEditor, KSOPrepEditor, NormalizeRadiusEditor, \
    MapToDataEditor, PyramidRescaleEditor, ImageNormalizeEditor, ReshapeEditor, sdo_norms, NormalizeEditor, \
    AIAPrepEditor, RemoveOffLimbEditor, StackEditor, soho_norms, RandomPatchEditor, NanEditor, LoadFITSEditor, \
    KSOFilmPrepEditor, ScaleEditor, ExpandDimsEditor, FeaturePatchEditor, EITCheckEditor, NormalizeExposureEditor, \
    PassEditor, BrightestPixelPatchEditor


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
        while len(samples) <  n_samples:
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
        while len(samples) <  n_samples:
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
                   NormalizeRadiusEditor(1024),
                   MapToDataEditor(),
                   PyramidRescaleEditor(1024 / resolution),
                   ImageNormalizeEditor(),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)


class KSOFilmDataset(BaseDataset):

    def __init__(self, path, resolution=256, ext="*.fts.gz", limit=None):
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))
        if limit:
            map_paths = random.sample(map_paths, limit)

        editors = [LoadFITSEditor(),
                   KSOFilmPrepEditor(),
                   NormalizeRadiusEditor(1024),
                   MapToDataEditor(),
                   PyramidRescaleEditor(1024 / resolution),
                   ImageNormalizeEditor(vmin=0, vmax=255),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)


class SDODataset(BaseDataset):

    def __init__(self, path, patch_shape=None):
        data_sets = [AIADataset(os.path.join(path, 'aia_171'), 171),
                     AIADataset(os.path.join(path, 'aia_193'), 193),
                     AIADataset(os.path.join(path, 'aia_211'), 211),
                     AIADataset(os.path.join(path, 'aia_304'), 304),
                     HMIDataset(os.path.join(path, 'hmi_mag'), 'mag')
                     ]
        editors = [StackEditor(data_sets)]
        if patch_shape is not None:
            editors.append(BrightestPixelPatchEditor(patch_shape))
        super().__init__(range(len(data_sets[0])), editors)


class SOHODataset(BaseDataset):

    def __init__(self, path, patch_shape=None):
        data_sets = [EITDataset(os.path.join(path, 'eit_171'), 171),
                     EITDataset(os.path.join(path, 'eit_195'), 195),
                     EITDataset(os.path.join(path, 'eit_284'), 284),
                     EITDataset(os.path.join(path, 'eit_304'), 304),
                     MDIDataset(os.path.join(path, 'mdi_mag'))
                     ]
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
                   NormalizeRadiusEditor(1024),
                   MapToDataEditor(),
                   PyramidRescaleEditor(1024 / resolution),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)


class MDIDataset(BaseDataset):

    def __init__(self, path, resolution=1024, ext='*.fits'):
        norm = soho_norms[6173]
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(1024),
                   RemoveOffLimbEditor(),
                   MapToDataEditor(),
                   NanEditor(),
                   PyramidRescaleEditor(1024 / resolution),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)


class AIADataset(BaseDataset):

    def __init__(self, path, wavelength, resolution=2048, ext='*.fits'):
        norm = sdo_norms[wavelength]
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(4096),
                   AIAPrepEditor(),
                   MapToDataEditor(),
                   PyramidRescaleEditor(4096 / resolution),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)


class HMIDataset(BaseDataset):

    def __init__(self, path, id, resolution=2048, ext='*.fits'):
        norm = sdo_norms[id]
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(4096),
                   RemoveOffLimbEditor(),
                   MapToDataEditor(),
                   NanEditor(),
                   PyramidRescaleEditor(4096 / resolution),
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
