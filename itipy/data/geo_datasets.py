from __future__ import annotations
import collections
import collections.abc

#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

import logging
import numpy as np
import xarray as xr
from typing import List, Union, Dict

from iti.data.editor import Editor
from iti.data.dataset import BaseDataset
from iti.data.geo_utils import get_split, get_list_filenames

# TODO: To be moved into ITI repo
class GeoDataset(BaseDataset):
    def __init__(
        self,
        data_dir: List[str],
        editors: List[Editor],
        splits_dict: Dict,
        ext: str="nc",
        limit: int=None,
        load_coords: bool=True,
        load_cloudmask: bool=True, 
        **kwargs
    ):
        """
        Initialize the GeoDataset class.

        Args:
            data_dir (List[str]): A list of directories containing the data files.
            editors (List[Editor]): A list of editors for data preprocessing.
            splits_dict (Dict, optional): A dictionary specifying the splits for the dataset. Defaults to None.
            ext (str, optional): The file extension of the data files. Defaults to "nc".
            limit (int, optional): The maximum number of files to load. Defaults to None.
            load_coords (bool, optional): Whether to load the coordinates. Defaults to True.
            load_cloudmask (bool, optional): Whether to load the cloud mask. Defaults to True.
            **kwargs: Additional keyword arguments.

        """
        self.data_dir = data_dir
        self.editors = editors
        self.splits_dict = splits_dict
        self.ext = ext
        self.limit = limit
        self.load_coords = load_coords
        self.load_cloudmask = load_cloudmask

        self.files = self.get_files()

        super().__init__(
            data=self.files,
            editors=self.editors,
            ext=self.ext,
            limit=self.limit,
            **kwargs
        )

    def get_files(self):
        # Get filenames from data_dir
        files = get_list_filenames(data_path=self.data_dir, ext=self.ext)
        # split files based on split criteria
        files = get_split(files=files, split_dict=self.splits_dict)
        return files

    def __len__(self):
        return len(self.files)
    
    def getIndex(self, data_dict, idx):
        # Attempt applying editors
        try:
            return self.convertData(data_dict)
        except Exception as ex:
            logging.error('Unable to convert %s: %s' % (self.files[idx], ex))
            raise ex

    def __getitem__(self, idx):
        data_dict = {}
        # Load dataset
        ds: xr.Dataset = xr.load_dataset(self.files[idx], engine="netcdf4")

        # Extract data
        data = ds.Rad.compute().to_numpy()
        data_dict["data"] = data
        # Extract wavelengths
        wavelengths = ds.band_wavelength.compute().to_numpy()
        data_dict["wavelengths"] = wavelengths

        # Extract coordinates
        if self.load_coords:
            latitude = ds.latitude.compute().to_numpy()
            longitude = ds.longitude.compute().to_numpy()
            coords = np.stack([latitude, longitude], axis=0)
            data_dict["coords"] = coords

        # Extract cloud mask
        if self.load_cloudmask:
            cloud_mask = ds.cloud_mask.compute().to_numpy()
            data_dict["cloud_mask"] = cloud_mask

        # Apply editors
        data, _ = self.getIndex(data_dict, idx)
        return data

        