import glob
import logging
import os
import random
from enum import Enum
from multiprocessing.pool import Pool
from typing import List
import numpy as np

from torch.utils.data import Dataset

from iti.data.editor import Editor, LoadMapEditor, KSOPrepEditor, NormalizeRadiusEditor, \
    MapToDataEditor, PyramidRescaleEditor, ImageNormalizeEditor, ReshapeEditor


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
        sample_data = random.sample(self.data, n_samples)
        return np.array(Pool(8).map(self.convertData, sample_data))

    def convertData(self, data):
        kwargs = {}
        for editor in self.editors:
            data, kwargs = editor.convert(data, **kwargs)
        return data

    def addEditor(self, editor):
        self.editors.append(editor)


class StorageDataset(Dataset):
    def __init__(self, dataset, store_dir):
        self.dataset = dataset
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        store_path = os.path.join(self.store_dir, '%d.npy' % idx)
        if os.path.exists(store_path):
            return np.load(store_path)
        data = self.dataset[idx]
        np.save(store_path, data)
        return data

    def sample(self, n_samples):
        indices = random.sample(range(len(self.dataset)), n_samples)
        return np.array(Pool(8).map(self.__getitem__, indices))

class KSODataset(BaseDataset):

    def __init__(self, path, resolution=256, ext="*.fts.gz", arcs_pp=1.02212, limit=None):
        map_paths = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))
        if limit:
            map_paths = random.sample(map_paths, limit)

        editors = [LoadMapEditor(),
                   KSOPrepEditor(),
                   NormalizeRadiusEditor(arcs_pp, 1024),
                   MapToDataEditor(),
                   PyramidRescaleEditor(1024 / resolution),
                   ImageNormalizeEditor(),
                   ReshapeEditor((1, resolution, resolution))]
        super().__init__(map_paths, editors=editors)


