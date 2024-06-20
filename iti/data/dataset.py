import gc
import glob
import logging
import os
import random
from collections import Iterable
from enum import Enum
from typing import List, Union

import numpy as np
from astropy.visualization import AsinhStretch
from dateutil.parser import parse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from iti.data.editor import Editor, LoadMapEditor, KSOPrepEditor, NormalizeRadiusEditor, \
    MapToDataEditor, ImageNormalizeEditor, ReshapeEditor, sdo_norms, NormalizeEditor, \
    AIAPrepEditor, RemoveOffLimbEditor, StackEditor, soho_norms, NanEditor, LoadFITSEditor, \
    KSOFilmPrepEditor, ScaleEditor, ExpandDimsEditor, FeaturePatchEditor, EITCheckEditor, NormalizeExposureEditor, \
    PassEditor, BrightestPixelPatchEditor, stereo_norms, LimbDarkeningCorrectionEditor, hinode_norms, gregor_norms, \
    LoadGregorGBandEditor, DistributeEditor, RecenterEditor, SECCHIPrepEditor, \
    SOHOFixHeaderEditor, PaddingEditor, hri_norm, proba2_norm, solo_norm


class Norm(Enum):
    """
    Enum for normalization types
    """
    CONTRAST = 'contrast'
    IMAGE = 'image'
    PEAK = 'adjusted'
    NONE = 'none'


class ArrayDataset(Dataset):
    """
    Dataset for numpy arrays

    Args:
        data (np.array): Data
        editors (List[Editor]): List of editors
        **kwargs: Additional arguments
    """

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
    """
    Base class for datasets

    Args:
        data (Union[str, list]): Data
        editors (List[Editor]): List of editors
        ext (str): File extension
        limit (int): Limit of samples
        months (list): List of months
        date_parser: Date parser
        **kwargs: Additional arguments
    """

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
    """
    Dataset for stacked data

    Args:
        data_sets (list): List of datasets
        limit (int): Limit of samples
        **kwargs: Additional arguments
    """

    def __init__(self, data_sets, limit=None, **kwargs):
        self.data_sets = data_sets

        editors = [StackEditor(data_sets)]
        super().__init__(list(range(len(data_sets[0]))), editors, limit=limit)

    def getId(self, idx):
        return os.path.basename(self.data_sets[0].data[idx]).split('.')[0]


class StorageDataset(Dataset):
    """
    Dataset for storing data to accelerate training

    Args:
        dataset (BaseDataset): Dataset
        store_dir (str): Storage directory
        ext_editors (list): List of editors
    """

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
            try:
                data = np.load(store_path, mmap_mode='r')
            except Exception as ex:
                logging.error('Unable to load %s: %s' % (store_path, ex))
                data = self.dataset[idx]
                np.save(store_path, data)
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
    """
    Get intersecting files from multiple directories

    Args:
        path (str): Path to directories
        dirs (list): List of directories
        months (list): List of months
        years (list): List of years
        n_samples (int): Number of samples
        ext (str): File extension
        basenames (list): List of basenames
        **kwargs: Additional arguments

    Returns:
        list: List of intersecting files
    """
    pattern = '*' if ext is None else '*' + ext
    if basenames is None:
        basenames = [
            [os.path.basename(path) for path in glob.glob(os.path.join(path, str(d), '**', pattern), recursive=True)]
            for d in dirs]
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
    """
    Dataset for SDO data

    Args:
        data: Data
        patch_shape (tuple): Patch shape
        wavelengths (list): List of wavelengths
        resolution (int): Resolution
        ext (str): File extension
        **kwargs: Additional arguments
    """

    def __init__(self, data, patch_shape=None, wavelengths=None, resolution=2048, ext='.fits', **kwargs):
        wavelengths = [171, 193, 211, 304, 6173, ] if wavelengths is None else wavelengths
        if isinstance(data, list):
            paths = data
        else:
            paths = get_intersecting_files(data, wavelengths, ext=ext, **kwargs)
        ds_mapping = {171: AIADataset, 193: AIADataset, 211: AIADataset, 304: AIADataset, 6173: HMIDataset}
        data_sets = [ds_mapping[wl_id](files, wavelength=wl_id, resolution=resolution, ext=ext)
                     for wl_id, files in zip(wavelengths, paths)]

        super().__init__(data_sets, **kwargs)
        if patch_shape is not None:
            self.addEditor(BrightestPixelPatchEditor(patch_shape))


class SOHODataset(StackDataset):
    """
    Dataset for SOHO data

    Args:
        data: Data
        patch_shape (tuple): Patch shape
        wavelengths (list): List of wavelengths
        resolution (int): Resolution
        ext (str): File extension
        **kwargs: Additional arguments
    """

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
    """
    Dataset for STEREO data

    Args:
        data: Data
        patch_shape (tuple): Patch shape
        resolution (int): Resolution
        **kwargs: Additional arguments
    """

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
    """
    Dataset for SOHO/EIT data

    Args:
        data: Data
        wavelength (int): Wavelength
        resolution (int): Resolution
        ext (str): File extension
        **kwargs: Additional arguments
    """

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
    """
    Dataset for SOHO/MDI data

    Args:
        data: Data
        wavelength (int): Wavelength
        resolution (int): Resolution
        ext (str): File extension
        **kwargs: Additional arguments
    """

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
    """
    Dataset for SDO/AIA data

    Args:
        data: Data
        wavelength (int): Wavelength
        resolution (int): Resolution
        ext (str): File extension
        calibration (str): Calibration type
        **kwargs: Additional arguments
    """

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
    """
    Dataset for SDO/HMI data

    Args:
        data: Data
        id (int): ID
        resolution (int): Resolution
        ext (str): File extension
        **kwargs: Additional arguments
    """

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
    """
    Dataset for SDO/HMI continuum data

    Args:
        data: Data
        patch_shape (tuple): Patch shape
        **kwargs: Additional arguments
    """

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
    """
    Dataset for SDO/ML data

    Args:
        data: Data
        patch_shape (tuple): Patch shape
        **kwargs: Additional arguments
    """

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
    """
    Dataset for Hinode data

    Args:
        data: Data
        scale (float): Scale
        wavelength (str): Wavelength
        **kwargs: Additional arguments
    """

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
    """
    Dataset for STEREO/SECCHI data

    Args:
        data: Data
        wavelength (int): Wavelength
        resolution (int): Resolution
        degradation (list): Degradation
        **kwargs: Additional arguments
    """

    def __init__(self, data, wavelength, resolution=1024, degradation=None, **kwargs):
        norm = stereo_norms[wavelength]

        editors = [LoadMapEditor(),
                   SECCHIPrepEditor(degradation),
                   NormalizeRadiusEditor(resolution),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(data, editors=editors, **kwargs)


class KSODataset(BaseDataset):
    """
    Dataset for KSO data

    Args:
        data: Data
        resolution (int): Resolution
        ext (str): File extension
        **kwargs: Additional arguments
    """

    def __init__(self, data: Union[str, list], resolution=256, ext=".fts.gz", **kwargs):
        editors = [LoadMapEditor(),
                   KSOPrepEditor(),
                   NormalizeRadiusEditor(resolution),
                   MapToDataEditor(),
                   ImageNormalizeEditor(0, 1000),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(data, editors=editors, ext=ext, **kwargs)


class KSOFlatDataset(BaseDataset):
    """
    Dataset for KSO flat data including limb darkening correction

    Args:
        data: Data
        resolution (int): Resolution
        ext (str): File extension
        date_parser: Date parser
        **kwargs: Additional arguments
    """

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
    """
    Dataset for KSO film data

    Args:
        data: Data
        resolution (int): Resolution
        ext (str): File extension
        date_parser: Date parser
        **kwargs: Additional arguments
    """

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
    """
    Dataset for GREGOR data

    Args:
        data: Data
        resolution (int): Resolution
        ext (str): File extension
        **kwargs: Additional arguments
    """

    def __init__(self, data, ext='.fts', **kwargs):
        norm = gregor_norms['gband']
        sub_editors = [MapToDataEditor(),
                       NanEditor(),
                       NormalizeEditor(norm),
                       ExpandDimsEditor()]
        editors = [LoadGregorGBandEditor(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)


class ZerosDataset(Dataset):
    """
    Dataset for zeros

    Args:
        length (int): Length
        resolution (int): Resolution
        **kwargs: Additional arguments
    """

    def __init__(self, length, resolution=1024, **kwargs):
        self.shape = (1, resolution, resolution)
        self.length = length

        super().__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return np.zeros(self.shape)


class EUIDataset(StackDataset):
    """
    Stacked Dataset for Solar Orbiter/EUI Full Sun Imager (FSI) data

    Args:
        data: Data
        patch_shape (tuple): Patch shape
        wavelengths (list): List of wavelengths
        resolution (int): Resolution
        ext (str): File extension
        **kwargs: Additional arguments
    """

    def __init__(self, data, patch_shape=None, wavelengths=None, resolution=1024, ext='.fits', **kwargs):
        wavelengths = ['eui-fsi174-image', 'eui-fsi304-image'] if wavelengths is None else wavelengths
        if isinstance(data, list):
            paths = data
        else:
            paths = get_intersecting_files(data, wavelengths, ext=ext, **kwargs)
        ds = {'eui-fsi174-image': FSIDataset, 'eui-fsi304-image': FSIDataset}
        data_sets = [ds[wl_id](files, wavelength=wl_id, resolution=resolution, ext=ext)
                     for wl_id, files in zip(wavelengths, paths)]

        super().__init__(data_sets, **kwargs)
        if patch_shape is not None:
            self.addEditor(BrightestPixelPatchEditor(patch_shape))


class FSIDataset(BaseDataset):
    """
    Dataset for Solar Orbiter/EUI Full Sun Imager (FSI) data

    Args:
        data: Data
        wavelength (str): Wavelength
        resolution (int): Resolution
        ext (str): File extension
        **kwargs: Additional arguments
    """

    def __init__(self, data, wavelength, resolution=1024, ext='.fits', **kwargs):
        norm = solo_norm[wavelength]

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution, fix_irradiance_with_distance=True),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(data, editors=editors, ext=ext, **kwargs)


class HRIDataset(BaseDataset):
    """
    Dataset for Solar Orbiter/EUI High Resolution Imager (HRI) data

    Args:
        data: Data
        resolution (int): Resolution
        ext (str): File extension
        **kwargs: Additional arguments
    """

    def __init__(self, data, resolution=4096, ext='.fits', **kwargs):
        norm = hri_norm[174]

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution=resolution, crop=True, rotate_north_up=False,
                                         fix_irradiance_with_distance=True),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ExpandDimsEditor()]
        super().__init__(data, editors=editors, ext=ext, **kwargs)


class SWAPDataset(BaseDataset):
    """
    Dataset for PROBA2/SWAP data

    Args:
        data: Data
        resolution (int): Resolution
        ext (str): File extension
        **kwargs: Additional arguments
    """

    def __init__(self, data, wavelength=174, patch_shape=None, resolution=1024, ext='.fits', **kwargs):
        norm = proba2_norm[wavelength]

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(data, editors=editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(BrightestPixelPatchEditor(patch_shape))
