import argparse
import os
import collections.abc
import shutil

import sys
# TODO: Fix path
sys.path.append("/home/anna.jungbluth/InstrumentToInstrument/")
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
#Now import hyper
import torch
import yaml
import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
#from lightning.pytorch.strategies import DataParallelStrategy

from iti.data.geo_datasets import GeoDataset
from iti.data.geo_editor import BandSelectionEditor, NanMaskEditor, CoordNormEditor, NanDictEditor, RadUnitEditor, ToTensorEditor, StackDictEditor, MeanStdNormEditor
from iti.data.editor import RandomPatchEditor
from iti.data.geo_utils import get_split, get_list_filenames, normalize

import warnings
warnings.filterwarnings('ignore')

from iti.callback import SaveCallback, PlotBAB, PlotABA
from iti.data.dataset import ITIDataModule
from iti.iti import ITIModule

from datetime import datetime

from loguru import logger

import xarray as xr


parser = argparse.ArgumentParser(description='Train MSG to GOES translations')
parser.add_argument('--config', 
                    default="/home/freischem/InstrumentToInstrument/config/msg_to_goes.yaml",
                    type=str, 
                    help='path to the config file.')

args = parser.parse_args()

with open(args.config, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Create timestamped directory within base_dir where normalisation, checkpoints (and prediction) are saved
base_dir = config['base_dir']
time_str = datetime.now().strftime("%Y%m%d-%H%M")
save_dir = os.path.join(base_dir, time_str)
os.makedirs(save_dir, exist_ok=True)

# Init Dataset
data_config = config['data']
msg_path = data_config['A_path']
goes_path = data_config['B_path']


splits_dict = { 
    "train": {"years": [2020], "months": [10], "days": list(range(1,20))},
    "val": {"years": [2020], "months": [10], "days": list(range(20,32))},
}

norm_config = config['normalization']
norm_ok = False
if 'norm_dir' in norm_config:
    logger.info(f"Loading normalization datasets saved in: {norm_config['norm_dir']}")
    try:
        goes_norm = xr.open_dataset(os.path.join(norm_config['norm_dir'], 'goes_norm.nc'))
        msg_norm = xr.open_dataset(os.path.join(norm_config['norm_dir'], 'msg_norm.nc'))
        norm_ok = True
    except:
        logger.warning(f"Unable to load normalization datasets from: {norm_config['norm_dir']}")

if not norm_ok:
    logger.info(f"Computing  means and stds for normalization...")

    # get list of files in training set
    goes_filenames = get_list_filenames(goes_path, ext='nc')
    msg_filenames = get_list_filenames(msg_path, ext='nc')

    goes_training_filenames = get_split(goes_filenames, splits_dict['train'])
    msg_training_filenames = get_split(msg_filenames, splits_dict['train'])

    # compute mean and std for list of training files
    goes_norm = normalize(goes_training_filenames)
    msg_norm = normalize(msg_training_filenames)

# save normalisations in current save directory
norm_dir = os.path.join(save_dir, 'normalization')
os.makedirs(norm_dir, exist_ok=True)
goes_norm.to_netcdf(os.path.join(norm_dir, 'goes_norm.nc'))
msg_norm.to_netcdf(os.path.join(norm_dir, 'msg_norm.nc'))

logger.info(f"Saved means and stds for normalization in {norm_dir}...")


goes_editors = [
    BandSelectionEditor(target_bands=[0.64, 3.89, 7.34, 9.61, 13.27]),
    # NanMaskEditor(key="data"), # Attaches nan_mask to the data dict
    # CoordNormEditor(key="coords"), # Normalizes lats/lons to [-1, 1]
    NanDictEditor(key="data", fill_value=0), # Replaces NaNs in data
    # NanDictEditor(key="coords", fill_value=0), # Replaces NaNs in coordinates
    # NanDictEditor(key="cloud_mask", fill_value=0), # Replaces NaNs in cloud_mask
    # RadUnitEditor(key="data"), TODO take into account for normalization if needed
    MeanStdNormEditor(norm_ds=goes_norm, key="data"),
    StackDictEditor(),
    ToTensorEditor(),
    RandomPatchEditor(patch_shape=(256, 256)),
]

msg_editors = [
    BandSelectionEditor(target_bands=[0.64, 3.92, 7.35, 9.66, 13.4]),
    # NanMaskEditor(key="data"), # Attaches nan_mask to the data dict
    # CoordNormEditor(key="coords"), # Normalizes lats/lons to [-1, 1]
    NanDictEditor(key="data", fill_value=0), # Replaces NaNs in data
    # NanDictEditor(key="coords", fill_value=0), # Replaces NaNs in coordinates
    # NanDictEditor(key="cloud_mask", fill_value=0), # Replaces NaNs in cloud_mask
    # RadUnitEditor(key="data"), TODO take into account for normalization if needed
    MeanStdNormEditor(norm_ds=msg_norm, key="data"),
    StackDictEditor(),
    ToTensorEditor(),
    RandomPatchEditor(patch_shape=(256, 256)),
]

logger.info(f"Instantiating datasets.")

msg_dataset = GeoDataset(
    data_dir=msg_path,
    editors=msg_editors,
    splits_dict=splits_dict['train'],
    load_coords=False,
    load_cloudmask=False,
)

msg_valid = GeoDataset(
    data_dir=msg_path,
    editors=msg_editors,
    splits_dict=splits_dict['val'],
    load_coords=False,
    load_cloudmask=False,
)

goes_dataset = GeoDataset(
    data_dir=goes_path,
    editors=goes_editors,
    splits_dict=splits_dict['train'],
    load_coords=False,
    load_cloudmask=False,
)

goes_valid = GeoDataset(
    data_dir=goes_path,
    editors=goes_editors,
    splits_dict=splits_dict['val'],
    load_coords=False,
    load_cloudmask=False,
)

data_module = ITIDataModule(msg_dataset, goes_dataset, msg_valid, goes_valid, **config['data'])

# setup logging

logger.info(f"Setting up WandB logging...")


logging_config = config['logging']
wandb_id = logging_config['wandb_id'] if 'wandb_id' in logging_config else None
log_model = logging_config['wandb_log_model'] if 'wandb_log_model' in logging_config else False

# Initialize wandb
run = wandb.init(project=logging_config['wandb_project'], 
                 name=logging_config['wandb_name'], 
                 entity=logging_config['wandb_entity'], 
                 id=wandb_id, 
                 dir=save_dir)
wandb_logger = WandbLogger(project=logging_config['wandb_project'], name=logging_config['wandb_name'], offline=False,
                           entity=logging_config['wandb_entity'], id=wandb_id, dir=save_dir, log_model=log_model)
# wandb_logger.experiment.config.update(config, allow_val_change=True)


logger.info(f"Initializing training steps...")

# Start training
module = ITIModule(**config['model'])

# setup save callbacks
logger.info(f"Initializing callbacks...")
checkpoint_dir = os.path.join(save_dir, 'checkpoints')
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, save_last=True, every_n_epochs=1, save_weights_only=False)
save_callback = SaveCallback(checkpoint_dir)

# setup plot callbacks
#prediction_dir = os.path.join(save_dir, 'prediction')
#os.makedirs(prediction_dir, exist_ok=True)
plot_callbacks = []


plot_settings_A = {"cmap": "viridris", "title": "MSG"} #, 'vmin': -1, 'vmax': 1}
plot_settings_B = {"cmap": "viridris", "title": "GOES-16"} #, 'vmin': -1, 'vmax': 1}


plot_callbacks += [PlotBAB(goes_valid.sample(1), module, plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]
plot_callbacks += [PlotABA(msg_valid.sample(1), module, plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]

n_gpus = torch.cuda.device_count()
n_cpus = os.cpu_count()

logger.info(f"Initializing Trainer...")
trainer = Trainer(
    max_epochs=int(config['training']['epochs']),
    fast_dev_run=False,
    logger=wandb_logger,
    devices=n_gpus if n_gpus > 0 else n_cpus,
    accelerator="gpu" if n_gpus >= 1 else "cpu",
    strategy='dp' if n_gpus > 1 else "auto",  # ddp breaks memory and wandb
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback, save_callback, *plot_callbacks],
)

logger.info(f"Starting training...")
trainer.fit(module, data_module, ckpt_path='last')
logger.info(f"Done...!")