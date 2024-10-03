from __future__ import annotations

import os

import dotenv
import hydra
import lightning.pytorch as pl
import numpy as np
import omegaconf
import torch
import wandb
er
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything

# MixedPrecisionPlugin
from lightning.pytorch.plugins import MixedPrecisionPlugin
from loguru import logger
from omegaconf import DictConfig

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

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

import sys

@hydra.main(
    version_base="1.1",
    config_path="configs/example-config",
    config_name="train.yaml",
)
def main(config: DictConfig):

    seed = config.seed
    logger.info(f"training with seed {seed}")
    seed_everything(seed, workers=True)

    # Create timestamped directory within base_dir where checkpoints (and prediction) are saved
    base_dir = config.base_dir
    time_str = datetime.now().strftime("%Y%m%d-%H%M")
    save_dir = os.path.join(base_dir, time_str)
    os.makedirs(save_dir, exist_ok=True)

    # Init Dataset
    A_path = config.A_data.A_path
    B_path = config.B_data.B_path

    # if normalization should be caculated on the fly
    if "normalization" in config:
        norm_config = config.normalization
        norm_ok = False
        if 'norm_dir' in norm_config:
            logger.info(f"Loading normalization datasets saved in: {norm_config['norm_dir']}")
            try:
                A_norm = xr.open_dataset(os.path.join(norm_config['norm_dir'], 'A_norm.nc'))
                B_norm = xr.open_dataset(os.path.join(norm_config['norm_dir'], 'B_norm.nc'))
                norm_ok = True
            except:
                logger.warning(f"Unable to load normalization datasets from: {norm_config['norm_dir']}")
        if not norm_ok:
            logger.info(f"Computing  means and stds for normalization...")

        # get list of files in training set
        A_filenames = get_list_filenames(A_path, ext='nc')
        B_filenames = get_list_filenames(B_path, ext='nc')

        A_training_filenames = get_split(A_filenames, splits_dict['train'])
        B_training_filenames = get_split(B_filenames, splits_dict['train'])

        # compute mean and std for list of training files
        A_norm = normalize(A_training_filenames)
        B_norm = normalize(B_training_filenames)

        # save normalisations in current save directory
        norm_dir = os.path.join(save_dir, 'normalization')
        os.makedirs(norm_dir, exist_ok=True)
        A_norm.to_netcdf(os.path.join(norm_dir, 'A_norm.nc'))
        B_norm.to_netcdf(os.path.join(norm_dir, 'B_norm.nc'))

        logger.info(f"Saved means and stds for normalization in {norm_dir}...")

    logger.info(f"Instantiating datasets.")

    A_train_dataset = hydra.utils.instantiate(config.A_data.A_train_dataset)
    A_val_dataset = hydra.utils.instantiate(config.A_data.A_val_dataset)

    B_train_dataset = hydra.utils.instantiate(config.B_data.B_train_dataset)
    B_val_dataset = hydra.utils.instantiate(config.B_data.B_val_dataset)

    data_module = ITIDataModule(
        A_train_ds=A_train_dataset,
        B_train_ds=B_train_dataset,
        A_valid_ds=A_val_dataset,
        B_valid_ds=B_val_dataset,
        **config['data'])

    # wandb logging
    logger.info(f"Setting up WandB logging...")
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    # get wandb tags
    tags = config.tags if "tags" in config else []
    if isinstance(tags, str):
        tags = tags.split()

    wandb_id = config['wandb_id'] if 'wandb_id' in config else None
    log_model = config['wandb_log_model'] if 'wandb_log_model' in config else False

    # Initialize wandb
    run = wandb.init(
        project=config['wandb_project'], 
        name=config['wandb_name'], 
        entity=config['wandb_entity'], 
        id=wandb_id, 
        dir=save_dir
        )
    wandb_logger = WandbLogger(
        project=config['wandb_project'], 
        name=config['wandb_name'], 
        offline=False,
        entity=config['wandb_entity'], 
        id=wandb_id, 
        dir=save_dir, 
        log_model=log_model,
        tags=tags
        )
    
    # Load model
    model = hydra.utils.instantiate(config.model)

    # setup save callbacks
    logger.info(f"Initializing callbacks...")
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, save_last=True, every_n_epochs=1, save_weights_only=False)
    save_callback = SaveCallback(checkpoint_dir)

    plot_callbacks = []

    plot_settings_A = config.A_data.A_plot_settings
    plot_settings_B = config.B_data.B_plot_settings

    plot_callbacks += [PlotBAB(B_val_dataset.sample(1), model, plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]
    plot_callbacks += [PlotABA(A_val_dataset.sample(1), model, plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]

    # Training details
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
    trainer.fit(model, data_module, ckpt_path='last')
    logger.info(f"Done...!")

if __name__ == "__main__":
    main()

# TODO: Create datasets with pre-defined editors

# goes_editors = [
#     BandSelectionEditor(target_bands=[0.64, 3.89, 7.34, 9.61, 13.27]),
#     # NanMaskEditor(key="data"), # Attaches nan_mask to the data dict
#     # CoordNormEditor(key="coords"), # Normalizes lats/lons to [-1, 1]
#     NanDictEditor(key="data", fill_value=0), # Replaces NaNs in data
#     # NanDictEditor(key="coords", fill_value=0), # Replaces NaNs in coordinates
#     # NanDictEditor(key="cloud_mask", fill_value=0), # Replaces NaNs in cloud_mask
#     # RadUnitEditor(key="data"), TODO take into account for normalization if needed
#     MeanStdNormEditor(norm_ds=goes_norm, key="data"),
#     StackDictEditor(),
#     ToTensorEditor(),
#     RandomPatchEditor(patch_shape=(256, 256)),
# ]

# msg_editors = [
#     BandSelectionEditor(target_bands=[0.64, 3.92, 7.35, 9.66, 13.4]),
#     # NanMaskEditor(key="data"), # Attaches nan_mask to the data dict
#     # CoordNormEditor(key="coords"), # Normalizes lats/lons to [-1, 1]
#     NanDictEditor(key="data", fill_value=0), # Replaces NaNs in data
#     # NanDictEditor(key="coords", fill_value=0), # Replaces NaNs in coordinates
#     # NanDictEditor(key="cloud_mask", fill_value=0), # Replaces NaNs in cloud_mask
#     # RadUnitEditor(key="data"), TODO take into account for normalization if needed
#     MeanStdNormEditor(norm_ds=msg_norm, key="data"),
#     StackDictEditor(),
#     ToTensorEditor(),
#     RandomPatchEditor(patch_shape=(256, 256)),
# ]