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
from iti.data.geo_editor import NanMaskEditor, CoordNormEditor, NanDictEditor, RadUnitEditor, ToTensorEditor, StackDictEditor
from iti.data.geo_utils import get_split, get_list_filenames, normalize

import warnings
warnings.filterwarnings('ignore')

from iti.callback import SaveCallback, PlotBAB, PlotABA
from iti.data.dataset import ITIDataModule
from iti.iti import ITIModule

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

base_dir = config['base_dir']
os.makedirs(base_dir, exist_ok=True)

# Init Dataset
data_config = config['data']
msg_path = data_config['A_path']
goes_path = data_config['B_path']

# splits_dict = { 
#     "train": {"years": None, "months": None, "days": None},
#     "val": {"years": None, "months": None, "days": None},
# }

splits_dict = { 
    "train": {"years": [2020], "months": [10], "days": list(range(1,20))},
    "val": {"years": [2020], "months": [10], "days": list(range(20,32))},
}

# TODO: Add data normalization!!!
# TODO remove satellite specific naming for hydra
# get list of files in training set
goes_filenames = get_list_filenames(goes_path, ext='nc')
msg_filenames = get_list_filenames(msg_path, ext='nc')

goes_training_filenames = get_split(goes_filenames, splits_dict['train'])
msg_training_filenames = get_split(msg_filenames, splits_dict['train'])

# compute mean and std for list of training files
goes_norm = normalize(goes_training_filenames)
msg_norm = normalize(msg_training_filenames)

# pass mean and std to Normalisation editor


# TODO: add band selection for miniset experiment

editors = [
    # BandSelectionEditor(target_bands=[0.47, 13.27]),
    # NanMaskEditor(key="data"), # Attaches nan_mask to the data dict
    # CoordNormEditor(key="coords"), # Normalizes lats/lons to [-1, 1]
    NanDictEditor(key="data", fill_value=0), # Replaces NaNs in data
    # NanDictEditor(key="coords", fill_value=0), # Replaces NaNs in coordinates
    # NanDictEditor(key="cloud_mask", fill_value=0), # Replaces NaNs in cloud_mask
    # RadUnitEditor(key="data"), TODO take into account for normalization if needed
    # TODO Normalization?
    StackDictEditor(),
    ToTensorEditor(),
]

"""
1. unit conversion
2. normalisation
"""



msg_dataset = GeoDataset(
    data_dir=msg_path,
    editors=editors,
    splits_dict=splits_dict['train'],
    load_coords=False,
    load_cloudmask=False,
)
msg_valid = GeoDataset(
    data_dir=msg_path,
    editors=editors,
    splits_dict=splits_dict['val'],
    load_coords=False,
    load_cloudmask=False,
)

goes_dataset = GeoDataset(
    data_dir=goes_path,
    editors=editors,
    splits_dict=splits_dict['train'],
    load_coords=False,
    load_cloudmask=False,
)

goes_valid = GeoDataset(
    data_dir=goes_path,
    editors=editors,
    splits_dict=splits_dict['val'],
    load_coords=False,
    load_cloudmask=False,
)

data_module = ITIDataModule(msg_dataset, goes_dataset, msg_valid, goes_valid, **config['data'])

# setup logging

logging_config = config['logging']
wandb_id = logging_config['wandb_id'] if 'wandb_id' in logging_config else None
log_model = logging_config['wandb_log_model'] if 'wandb_log_model' in logging_config else False

# Initialize wandb
run = wandb.init(project=logging_config['wandb_project'], 
                 name=logging_config['wandb_name'], 
                 entity=logging_config['wandb_entity'], 
                 id=wandb_id, 
                 dir=config['base_dir'])
wandb_logger = WandbLogger(project=logging_config['wandb_project'], name=logging_config['wandb_name'], offline=False,
                           entity=logging_config['wandb_entity'], id=wandb_id, dir=config['base_dir'], log_model=log_model)
# wandb_logger.experiment.config.update(config, allow_val_change=True)

# Start training
module = ITIModule(**config['model'])

# setup save callbacks
checkpoint_callback = ModelCheckpoint(dirpath=base_dir, save_last=True, every_n_epochs=1, save_weights_only=False)
save_callback = SaveCallback(base_dir)

# setup plot callbacks
#prediction_dir = os.path.join(base_dir, 'prediction')
#os.makedirs(prediction_dir, exist_ok=True)
plot_callbacks = []
plot_callbacks += [PlotBAB(goes_valid.sample(1), module, plot_settings_A=None, plot_settings_B=None)]
plot_callbacks += [PlotABA(msg_valid.sample(1), module, plot_settings_A=None, plot_settings_B=None)]

n_gpus = torch.cuda.device_count()
n_cpus = os.cpu_count()

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

trainer.fit(module, data_module, ckpt_path='last')