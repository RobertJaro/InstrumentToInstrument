import argparse
import os
from random import sample

import pandas as pd
import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from itipy.callback import SaveCallback, PlotBAB, PlotABA
from itipy.data.dataset import StorageDataset, HinodeDataset, \
    HMIContinuumDataset
from itipy.data.data_module import ITIDataModule
from itipy.data.editor import RandomPatchEditor
from itipy.iti import ITIModule

parser = argparse.ArgumentParser(description='Train HMI-To-Hinode translations')
parser.add_argument('--config', type=str, help='path to the config file.')

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
hmi_path = data_config['A_path']
hmi_converted_path = data_config['converted_A_path']
hinode_file_list = data_config['B_path']
hinode_converted_path = data_config['converted_B_path']

test_months = [11, 12]
train_months = list(range(2, 10))

df = pd.read_csv(hinode_file_list, index_col=False, parse_dates=['date'])

train_df = df[df.date.dt.month.isin(train_months)]
features = train_df[train_df.classification == 'feature']
quiet = train_df[train_df.classification == 'quiet']
limb = train_df[train_df.classification == 'limb']
hinode_train_files = list(features.file) + list(limb.file) + sample(list(quiet.file), len(features) + len(limb))

test_df = df[df.date.dt.month.isin(test_months)]
features = train_df[train_df.classification == 'feature']
quiet = train_df[train_df.classification == 'quiet']
limb = train_df[train_df.classification == 'limb']
hinode_test_files = list(features.file) + list(limb.file) + sample(list(quiet.file), len(features) + len(limb))

# Init Dataset
hmi_train = HMIContinuumDataset(hmi_path, (512, 512), months=train_months, ext='.fits')
hmi_train = StorageDataset(hmi_train, hmi_converted_path, ext_editors=[RandomPatchEditor((160, 160))])

hmi_valid = HMIContinuumDataset(hmi_path, (512, 512), months=test_months, ext='.fits')
hmi_valid = StorageDataset(hmi_valid, hmi_converted_path, ext_editors=[RandomPatchEditor((160, 160))])

hinode_train = HinodeDataset(hinode_train_files)
hinode_train = StorageDataset(hinode_train, hinode_converted_path,
                              ext_editors=[RandomPatchEditor((640, 640))])

hinode_valid = HinodeDataset(hinode_test_files)
hinode_valid = StorageDataset(hinode_valid, hinode_converted_path, ext_editors=[RandomPatchEditor((640, 640))])

data_module = ITIDataModule(hmi_train, hinode_train, hmi_valid, hinode_valid, **config['data'])

plot_settings_A = [
    {"cmap": "gray", "title": "HMI Continuum", 'vmin': -1, 'vmax': 1, },
]
plot_settings_B = [
    {"cmap": "gray", "title": "Hinode Continuum", 'vmin': -1, 'vmax': 1},
]

# setup logging
wandb_logger = WandbLogger(**config['logging'], dir=config['base_dir'])
wandb_logger.experiment.config.update(config, allow_val_change=True)

# Start training
module = ITIModule(**config['model'])

# setup save callbacks
checkpoint_callback = ModelCheckpoint(dirpath=base_dir, save_last=True, every_n_epochs=1)
save_callback = SaveCallback(base_dir)

# setup plot callbacks
prediction_dir = os.path.join(base_dir, 'prediction')
os.makedirs(prediction_dir, exist_ok=True)
plot_callbacks = []
plot_callbacks += [
    PlotBAB(hinode_valid.sample(4), module, plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]
plot_callbacks += [
    PlotABA(hmi_valid.sample(4), module, plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]

n_gpus = torch.cuda.device_count()
trainer = Trainer(max_epochs=int(config['training']['epochs']),
                  logger=wandb_logger,
                  devices=n_gpus if n_gpus > 0 else None,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                  num_sanity_val_steps=-1,
                  callbacks=[checkpoint_callback, save_callback, *plot_callbacks], )

trainer.fit(module, data_module, ckpt_path='last')
