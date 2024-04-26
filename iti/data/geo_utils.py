from typing import Optional, List, Union, Tuple
from omegaconf import DictConfig
from datetime import datetime
import pandas as pd
from loguru import logger
import glob, os
import numpy as np
import pandas as pd

def split_train_val(files: List, split_spec: DictConfig) -> Tuple[List, List]:
    """
    Split files into training and validation sets based on dataset specification.

    Args:
        files (List): A list of files to be split.
        split_spec (DictConfig): A dictionary-like object containing the dataset specification.

    Returns:
        Tuple[List, List]: A tuple containing two lists: the training set and the validation set.
    """
    if "train" not in split_spec.keys() or "val" not in split_spec.keys():
        raise ValueError("split_spec must contain 'train' and 'val' keys")
    
    train_files = get_split(files, split_spec["train"])
    val_files = get_split(files, split_spec["val"])

    return train_files, val_files
    
    
def get_split(files: List, 
              split_dict: DictConfig) -> Tuple[List, List]:
    """
    Split files based on dataset specification.

    Args:
        files (List): A list of files to be split.
        split_dict (DictConfig): A dictionary-like object containing the dataset specification.

    Returns:
        Tuple[List, List]: A tuple containing two lists: the training set and the validation set.
    """
    # Extract dates from filenames
    filenames = [file.split("/")[-1] for file in files]
    dates = get_dates_from_files(filenames)
    # Convert to dataframe for easier manipulation
    df = pd.DataFrame({"filename": filenames, "files": files, "date": dates})

    # Check if years, months, and days are specified
    if "years" not in split_dict.keys() or split_dict["years"] is None:
        logger.info("No years specified for split. Using all years.")
        split_dict["years"] = df.date.dt.year.unique().tolist()
    if "months" not in split_dict.keys() or split_dict["months"] is None:
        logger.info("No months specified for split. Using all months.")
        split_dict["months"] = df.date.dt.month.unique().tolist()
    if "days" not in split_dict.keys() or split_dict["days"] is None:
        logger.info("No days specified for split. Using all days.")
        split_dict["days"] = df.date.dt.day.unique().tolist()

    # Determine conditions specified split
    condition = (df.date.dt.year.isin(split_dict["years"])) & \
                (df.date.dt.month.isin(split_dict["months"])) & \
                (df.date.dt.day.isin(split_dict["days"]))
        
    # Extract filenames based on conditions
    split_files = df[condition].files.tolist()

    # Check if files are allocated properly
    if len(split_files) == 0:
        raise ValueError("No files found. Check split specification.")
    
    return split_files

def get_date_from_file(filename: str) -> datetime:
    """
    Extract date from filename.

    Args:
        filenames (List[str]): A list of filenames.

    Returns:
        List[str]: A list of dates extracted from the filenames.
    """
    date = datetime.strptime(filename.split("_")[0], "%Y%m%d%H%M%S")
    return date

def get_dates_from_files(filenames: List[str]) -> List[datetime]:
    """
    Extract dates from a list of filenames.

    Args:
        filenames (List[str]): A list of filenames.

    Returns:
        List[str]: A list of dates extracted from the filenames.
    """
    dates = [datetime.strptime(filename.split("_")[0], "%Y%m%d%H%M%S") for filename in filenames]
    return dates

def get_list_filenames(data_path: str="./", ext: str="*"):
    """
    Loads a list of file names within a directory.

    Args:
        data_path (str, optional): The directory path to search for files. Defaults to "./".
        ext (str, optional): The file extension to filter the search. Defaults to "*".

    Returns:
        List[str]: A sorted list of file names matching the given extension within the directory.
    """
    pattern = f"*{ext}"
    return sorted(glob.glob(os.path.join(data_path, "**", pattern), recursive=True))

def get_files(datasets_spec: DictConfig, ext=".nc"):
    """
    Get a list of filenames based on the provided datasets specification.

    Args:
        datasets_spec (DictConfig): The datasets specification containing the path and extension.
        ext (str, optional): The file extension to filter the search. Defaults to ".nc".

    Returns:
        List[str]: A list of filenames.

    """
    data_path = datasets_spec.data_path
    return get_list_filenames(data_path=data_path, ext=ext)


def convert_units(data: np.array, wavelengths: np.array) -> np.array:
    """
    Function to convert units from mW/m^2/sr/cm^-1 to W/m^2/sr/um in numpy array.
    Acts on each band separately.
    
    Parameters:
        data (np.array): The input data to be converted.
        wavelengths (np.array): The wavelengths of the input data.
        
    Returns:
        np.array: The converted data.
    """
    assert len(data) == len(wavelengths)
    corrected_data = []
    for i, wvl in enumerate(wavelengths):
        corr_data = data[i] * 0.001 # to convert mW to W
        corr_data = corr_data * 10000 / wvl**2 # to convert cm^-1 to um
        corrected_data.append(corr_data)
    return np.stack(corrected_data, axis=0)



