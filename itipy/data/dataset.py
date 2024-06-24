import gc
import glob
import logging
import os
import random
import warnings
from collections import Iterable
from enum import Enum
from typing import List, Union

import numpy as np
from astropy.visualization import AsinhStretch
from dateutil.parser import parse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from itipy.data.editor import Editor, LoadMapEditor, KSOPrepEditor, NormalizeRadiusEditor, \
    MapToDataEditor, ImageNormalizeEditor, ReshapeEditor, sdo_norms, NormalizeEditor, \
    AIAPrepEditor, RemoveOffLimbEditor, StackEditor, soho_norms, NanEditor, LoadFITSEditor, \
    KSOFilmPrepEditor, ScaleEditor, ExpandDimsEditor, FeaturePatchEditor, EITCheckEditor, NormalizeExposureEditor, \
    PassEditor, BrightestPixelPatchEditor, stereo_norms, LimbDarkeningCorrectionEditor, hinode_norms, gregor_norms, \
    LoadGregorGBandEditor, DistributeEditor, RecenterEditor, AddRadialDistanceEditor, SECCHIPrepEditor, \
    SOHOFixHeaderEditor, PaddingEditor


class Norm(Enum):
    CONTRAST = 'contrast'
    IMAGE = 'image'
    PEAK = 'adjusted'
    NONE = 'none'

class ArrayDataset(Dataset):

    def __init__(self, data, editors: List[Editor], **kwargs):
        self.data = data
        self.editors = editors

        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, _ = self.getIndex(idx)
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

    def getIndex(self, idx):
        try:
            return self.convertData(self.data[idx])
        except Exception as ex:
            logging.error('Unable to convert %s: %s' % (self.data[idx], ex))
            raise ex

    def getId(self, idx):
        return str(idx)

    def convertData(self, data):
        kwargs = {}
        for editor in self.editors:
            data, kwargs = editor.convert(data, **kwargs)
        return data, kwargs

    def addEditor(self, editor):
        self.editors.append(editor)


class BaseDataset(Dataset):

    def __init__(self, data: Union[str, list], editors: List[Editor], ext: str = None, limit: int = None,
                 months: list = None, date_parser=None, **kwargs):
        if isinstance(data, str):
            pattern = '*' if ext is None else '*' + ext
            data = sorted(glob.glob(os.path.join(data, "**", pattern), recursive=True))
        assert isinstance(data, Iterable), 'Dataset requires list of samples or path to files!'
        if months:  # assuming filename is parsable datetime
            if date_parser is None:
                date_parser = lambda f: parse(os.path.basename(f).split('.')[0])
            data = [d for d in data if date_parser(d).month in months]
        if limit is not None:
            data = random.sample(list(data), limit)
        self.data = data
        self.editors = editors

        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, _ = self.getIndex(idx)
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

    def getIndex(self, idx):
        try:
            return self.convertData(self.data[idx])
        except Exception as ex:
            logging.error('Unable to convert %s: %s' % (self.data[idx], ex))
            raise ex

    def getId(self, idx):
        return os.path.basename(self.data[idx]).split('.')[0]

    def convertData(self, data):
        kwargs = {}
        for editor in self.editors:
            data, kwargs = editor.convert(data, **kwargs)
        return data, kwargs

    def addEditor(self, editor):
        self.editors.append(editor)


class StackDataset(BaseDataset):

    def __init__(self, data_sets, limit=None, **kwargs):
        self.data_sets = data_sets

        editors = [StackEditor(data_sets)]
        super().__init__(list(range(len(data_sets[0]))), editors, limit=limit)

    def getId(self, idx):
        return os.path.basename(self.data_sets[0].data[idx]).split('.')[0]


class StorageDataset(Dataset):
    def __init__(self, dataset: BaseDataset, store_dir, ext_editors=[]):
        self.dataset = dataset
        self.store_dir = store_dir
        self.ext_editors = ext_editors
        os.makedirs(store_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        id = self.dataset.getId(idx)
        store_path = os.path.join(self.store_dir, '%s.npy' % id)
        if os.path.exists(store_path):
            data = np.load(store_path, mmap_mode='r')
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
                samples.append(next(it).detach().numpy())
            except Exception as ex:
                logging.error(str(ex))
                continue
        del it
        return np.concatenate(samples)

    def convert(self, n_worker):
        it = DataLoader(self, batch_size=1, shuffle=False, num_workers=n_worker).__iter__()
        for i in tqdm(range(len(self.dataset))):
            try:
                next(it)
                gc.collect()
            except StopIteration:
                return
            except Exception as ex:
                logging.error('Invalid data: %s' % self.dataset.data[i])
                logging.error(str(ex))
                continue


def get_intersecting_files(path, dirs, months=None, years=None, n_samples=None, ext=None, basenames=None, **kwargs):
    pattern = '*' if ext is None else '*' + ext
    if basenames is None:
        basenames = [[os.path.basename(path) for path in glob.glob(os.path.join(path, str(d), '**', pattern), recursive=True)] for d in dirs]
        basenames = list(set(basenames[0]).intersection(*basenames))
    if months:  # assuming filename is parsable datetime
        basenames = [bn for bn in basenames if parse(bn.split('.')[0]).month in months]
    if years:  # assuming filename is parsable datetime
        basenames = [bn for bn in basenames if parse(bn.split('.')[0]).year in years]
    basenames = sorted(list(basenames))
    if n_samples:
        basenames = basenames[::len(basenames) // n_samples]
    return [[os.path.join(path, str(dir), b) for b in basenames] for dir in dirs]


class SDODataset(StackDataset):

    def __init__(self, data, patch_shape=None, resolution=2048, ext='.fits', **kwargs):
        if isinstance(data, list):
            paths = data
        else:
            paths = get_intersecting_files(data, ['171', '193', '211', '304', '6173'], ext=ext, **kwargs)
        data_sets = [AIADataset(paths[0], 171, resolution=resolution, **kwargs),
                     AIADataset(paths[1], 193, resolution=resolution, **kwargs),
                     AIADataset(paths[2], 211, resolution=resolution, **kwargs),
                     AIADataset(paths[3], 304, resolution=resolution, **kwargs),
                     HMIDataset(paths[4], 'mag', resolution=resolution)
                     ]
        super().__init__(data_sets, **kwargs)
        if patch_shape is not None:
            self.addEditor(BrightestPixelPatchEditor(patch_shape))


class SOHODataset(StackDataset):

    def __init__(self, data, patch_shape=None, resolution=1024, ext='.fits', wavelengths=None, **kwargs):
        wavelengths = [171, 195, 284, 304, 'mag', ] if wavelengths is None else wavelengths
        if isinstance(data, list):
            paths = data
        else:
            paths = get_intersecting_files(data, wavelengths, ext=ext, **kwargs)

        ds = {171: EITDataset, 195: EITDataset, 284: EITDataset, 304: EITDataset, 'mag': MDIDataset}
        data_sets = [ds[wl_id](files, wavelength=wl_id, resolution=resolution, ext=ext)
                     for wl_id, files in zip(wavelengths, paths)]

        super().__init__(data_sets, **kwargs)
        if patch_shape is not None:
            self.addEditor(BrightestPixelPatchEditor(patch_shape))


class STEREODataset(StackDataset):

    def __init__(self, data, patch_shape=None, resolution=1024, **kwargs):
        if isinstance(data, list):
            paths = data
        else:
            paths = get_intersecting_files(data, ['171', '195', '284', '304'], **kwargs)
        data_sets = [SECCHIDataset(paths[0], 171, resolution=resolution),
                     SECCHIDataset(paths[1], 195, resolution=resolution),
                     SECCHIDataset(paths[2], 284, resolution=resolution),
                     SECCHIDataset(paths[3], 304, resolution=resolution, degradation=[-9.42497209e-05, 2.27153104e+00]),
                     ]
        super().__init__(data_sets, **kwargs)
        if patch_shape is not None:
            self.addEditor(BrightestPixelPatchEditor(patch_shape))


class EITDataset(BaseDataset):

    def __init__(self, data, wavelength, resolution=1024, ext='.fits', **kwargs):
        norm = soho_norms[wavelength]

        editors = [LoadMapEditor(),
                   EITCheckEditor(),
                   SOHOFixHeaderEditor(),
                   NormalizeRadiusEditor(resolution),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(data, editors=editors, ext=ext, **kwargs)


class MDIDataset(BaseDataset):

    def __init__(self, data, resolution=1024, ext='.fits', **kwargs):
        norm = soho_norms[6173]
        editors = [LoadMapEditor(),
                   SOHOFixHeaderEditor(),
                   NormalizeRadiusEditor(resolution),
                   RemoveOffLimbEditor(),
                   MapToDataEditor(),
                   NanEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(data, editors=editors, ext=ext, **kwargs)


class AIADataset(BaseDataset):

    def __init__(self, data, wavelength, resolution=2048, ext='.fits', calibration='auto', **kwargs):
        norm = sdo_norms[wavelength]

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution),
                   AIAPrepEditor(calibration=calibration),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(data, editors=editors, ext=ext, **kwargs)


class HMIDataset(BaseDataset):

    def __init__(self, data, id, resolution=2048, ext='.fits', **kwargs):
        norm = sdo_norms[id]

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution),
                   RemoveOffLimbEditor(),
                   MapToDataEditor(),
                   PaddingEditor((resolution, resolution)),  # fix field-of-view of subframe
                   NanEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(data, editors=editors, ext=ext, **kwargs)


class HMIContinuumDataset(BaseDataset):

    def __init__(self, data, patch_shape=None, **kwargs):
        norm = sdo_norms['continuum']

        editors = [LoadMapEditor(),
                   RecenterEditor(),
                   ScaleEditor(0.6),
                   FeaturePatchEditor(patch_shape) if patch_shape is not None else PassEditor(),
                   MapToDataEditor(),
                   NanEditor(),
                   NormalizeEditor(norm),
                   ExpandDimsEditor()]
        super().__init__(data, editors=editors, **kwargs)


class SDOMLDataset(BaseDataset):

    def __init__(self, data, wavelength, resolution=2048, ext='.fits', **kwargs):
        norm = sdo_norms[wavelength]

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution),
                   AIAPrepEditor(),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(data, editors=editors, ext=ext, **kwargs)

class HinodeDataset(BaseDataset):

    def __init__(self, data, scale=0.15, wavelength='continuum', **kwargs):
        norm = hinode_norms[wavelength]

        editors = [LoadMapEditor(),
                   ScaleEditor(scale),
                   NormalizeExposureEditor(),
                   MapToDataEditor(),
                   NanEditor(),
                   NormalizeEditor(norm),
                   ExpandDimsEditor()]
        super().__init__(data, editors=editors, **kwargs)


class SECCHIDataset(BaseDataset):

    def __init__(self, data, wavelength, resolution=1024, degradation=None,**kwargs):
        norm = stereo_norms[wavelength]

        editors = [LoadMapEditor(),
                   SECCHIPrepEditor(degradation),
                   NormalizeRadiusEditor(resolution),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(data, editors=editors, **kwargs)


class KSODataset(BaseDataset):

    def __init__(self, data: Union[str, list], resolution=256, ext=".fts.gz", **kwargs):
        editors = [LoadMapEditor(),
                   KSOPrepEditor(),
                   NormalizeRadiusEditor(resolution),
                   MapToDataEditor(),
                   ImageNormalizeEditor(0, 1000),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(data, editors=editors, ext=ext, **kwargs)


class KSOFlatDataset(BaseDataset):

    def __init__(self, data, resolution=256, ext=".fts.gz", date_parser=None, **kwargs):
        editors = [LoadMapEditor(),
                   KSOPrepEditor(),
                   NormalizeRadiusEditor(resolution, 0),
                   LimbDarkeningCorrectionEditor(),
                   MapToDataEditor(),
                   ImageNormalizeEditor(0.65, 1.5, stretch=AsinhStretch(0.5)),
                   NanEditor(-1),
                   ReshapeEditor((1, resolution, resolution))]
        if date_parser is None:
            date_parser = lambda f: parse(os.path.basename(f)[14:-7].replace('_', 'T'))
        super().__init__(data, editors=editors, ext=ext, date_parser=date_parser, **kwargs)


class KSOFilmDataset(BaseDataset):

    def __init__(self, data, resolution=256, ext=".fts.gz", date_parser=None, **kwargs):
        editors = [LoadFITSEditor(),
                   KSOFilmPrepEditor(),
                   NormalizeRadiusEditor(resolution, 0),
                   LimbDarkeningCorrectionEditor(),
                   MapToDataEditor(),
                   ImageNormalizeEditor(0.39, 1.94, stretch=AsinhStretch(0.5)),
                   NanEditor(-1),
                   ReshapeEditor((1, resolution, resolution))]
        if date_parser is None:
            date_parser = lambda f: parse(os.path.basename(f)[-22:-7].replace('_', 'T'))
        super().__init__(data, editors=editors, ext=ext, date_parser=date_parser, **kwargs)


class GregorDataset(BaseDataset):

    def __init__(self, data, ext='.fts', **kwargs):
        norm = gregor_norms['gband']
        sub_editors = [MapToDataEditor(),
                       NanEditor(),
                       NormalizeEditor(norm),
                       ExpandDimsEditor()]
        editors = [LoadGregorGBandEditor(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)


class ZerosDataset(Dataset):

    def __init__(self, length, resolution=1024, **kwargs):
        self.shape = (1, resolution, resolution)
        self.length = length

        super().__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return np.zeros(self.shape)
