import torch
import numpy as np

from itipy.data.editor import Editor
from itipy.data.geo_utils import convert_units

class BandOrderEditor(Editor):
    """
    Reorders bands in data dictionary.
    """

    def __init__(self, target_order, key="data"):
        """
        Args:
            target_order (list): Order of bands
            key (str): Key in dictionary to apply transformation
        """
        self.target_order = target_order
        self.key = key

    def call(self, data_dict, **kwargs):
        source_order = data_dict["wavelengths"]
        assert len(source_order) == len(self.target_order), "Length of source and target wavelengths must match."
        # Get indexes of bands to select
        indexes = [np.where(source_order == wvl)[0][0] for wvl in self.target_order]
        # Extract data
        data = data_dict[self.key]
        # Subselect bands
        data = data[indexes]
        # Update dictionary
        data_dict[self.key] = data
        data_dict["wavelengths"] = np.array(self.target_order)
        return data_dict

class BandSelectionEditor(Editor):
    """
    Selects a subset of available bands from data dictionary
    """
    def __init__(self, target_bands, key="data"):
        """
        Args:
            target_bands (list): List of bands to select
            key (str): Key in dictionary to apply transformation
        """
        self.target_bands = target_bands
        self.key = key

    def call(self, data_dict, **kwargs):
        source_bands = data_dict["wavelengths"]
        # Get indexes of bands to select
        indexes = [np.where(source_bands == wvl)[0][0] for wvl in self.target_bands]
        # Extract data
        data = data_dict[self.key]
        # Subselect bands
        data = data[indexes]
        assert data.shape[0] == len(self.target_bands)
        # Update dictionary
        data_dict[self.key] = data
        data_dict["wavelengths"] = np.array(self.target_bands)
        return data_dict

class NanMaskEditor(Editor):
    """
    Returns mask for NaN values in data dictionary
    """
    def __init__(self, key="data"):
        self.key = key
    def call(self, data_dict, **kwargs):
        data = data_dict[self.key]
        # Check if any band contains NaN values
        mask = np.isnan(data).any(axis=0)
        mask = mask.astype(int)
        # Update dictionary
        data_dict["nan_mask"] = mask
        return data_dict

class NanDictEditor(Editor):
    """
    Removes NaN values from data dictionary.
    Can also be used to replace NaN values of coordinates to remove off limb data.
    """
    def __init__(self, key="data", fill_value=0):
        self.key = key
        self.fill_value = fill_value
    def call(self, data_dict, **kwargs):
        data = data_dict[self.key]
        # Replace NaN values
        data = np.nan_to_num(data, nan=self.fill_value)
        # Update dictionary
        data_dict[self.key] = data
        return data_dict
    
class CoordNormEditor(Editor):
    """
    Normalize latitude and longitude coordinates
    """
    def __init__(self, key="coords"):
        self.key = key
    def call(self, data_dict, **kwargs):
        lats, lons = data_dict["coords"]
        # Normalize latitude and longitude to range [-1, 1]
        lats = lats/90
        lons = lons/180
        # Update dictionary
        data_dict["coords"] = np.stack([lats, lons], axis=0)
        return data_dict
     
class RadUnitEditor(Editor):
    """
    Convert radiance values from mW/m^2/sr/cm^-1 to W/m^2/sr/um
    """
    def __init__(self, key="data"):
        self.key = key
    def call(self, data_dict, **kwargs):
        data = data_dict[self.key]
        wavelengths = data_dict["wavelengths"]
        # Convert units
        data = convert_units(data, wavelengths)
        # Update dictionary
        data_dict[self.key] = data
        return data_dict
    
class StackDictEditor(Editor):
    """
    Stack data dictionary into a single array
    """
    def __init__(self, allowed_keys=["data", "cloud_mask", "nan_mask", "coords"], axis=0):
        self.allowed_keys = allowed_keys
        self.axis = axis
    def call(self, data_dict, **kwargs):
        # Select keys
        self.keys = [key for key in self.allowed_keys if key in data_dict.keys()]
        # Select data
        data = []
        for key in self.keys:
            values = data_dict[key]
            if len(values.shape) == 2:
                values = np.expand_dims(values, axis=self.axis)
            data.append(values)
        # Stack data
        data = np.concatenate(data, axis=self.axis)
        # Return numpy array
        return data
    
class ToTensorEditor(Editor):
    """
    Convert numpy array to PyTorch tensor
    """
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
    def call(self, data, **kwargs):
        # Convert to tensor
        tensor = torch.as_tensor(data, dtype=self.dtype)
        return tensor

class MeanStdNormEditor(Editor):
    """
    Normalise each band in the data using the mean and std from the norm_ds.
    """
    def __init__(self, norm_ds, key="data"):
        """
        Args:
            norm_ds (xarray.Dataset): Dataset with normalization values (mean and std)
            key (str): Key in dictionary to apply transformation
        """
        self.key = key
        self.norm = norm_ds

    def call(self, data_dict, **kwargs):
        data = data_dict[self.key]
        # use wavelengths and only normalise the bands that we have in the data
        data_wavelengths = data_dict["wavelengths"]
        # Get indeces of bands to select
        indeces = [np.where(self.norm.band_wavelength == wvl)[0][0] for wvl in data_wavelengths]
        
        # extract relevant means and stds
        means = self.norm['mean'][indeces].values
        stds = self.norm['std'][indeces].values

        # check that number of channels equals number of means & stds
        assert data.shape[0] == means.shape[0]
        assert data.shape[0] == stds.shape[0]

        # apply normalization
        data = (data - means[:, None, None]) / stds[:, None, None]
        
        # Update dictionary
        data_dict[self.key] = data
        return data_dict



