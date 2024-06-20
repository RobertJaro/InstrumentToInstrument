import argparse
import os
import collections.abc
import shutil

#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
#Now import hyper
import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
#from lightning.pytorch.strategies import DataParallelStrategy
from sunpy.visualization.colormaps import cm

from iti.callback import SaveCallback, PlotBAB, PlotABA
from iti.data.dataset import HRIDataset, StorageDataset, AIADataset, ITIDataModule
from iti.data.editor import RandomPatchEditor, BrightestPixelPatchEditor
from iti.iti import ITIModule

parser = argparse.ArgumentParser(description='Train FSI-To-HRI translations')
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
aia_path = data_config['A_path']
aia_converted_path = data_config['converted_A_path']
hri_path = data_config['B_path']
hri_converted_path = data_config['converted_B_path']

test_months = [9]
train_months = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]

hri_dataset = HRIDataset(hri_path, months=train_months)
hri_dataset = StorageDataset(hri_dataset,
                             hri_converted_path,
                             ext_editors=[RandomPatchEditor((512, 512))])

aia_dataset = AIADataset(aia_path, wavelength=171, months=train_months)
aia_dataset = StorageDataset(aia_dataset, aia_converted_path,
                                ext_editors=[RandomPatchEditor((128, 128))])

hri_valid = StorageDataset(HRIDataset(hri_path, months=test_months, limit=40),
                           hri_converted_path, ext_editors=[RandomPatchEditor((512, 512))])
aia_valid = StorageDataset(AIADataset(aia_path, wavelength=171, months=test_months, limit=40),
                              aia_converted_path, ext_editors=[RandomPatchEditor((128, 128))])

data_module = ITIDataModule(aia_dataset, hri_dataset, aia_valid, hri_valid, **config['data'])

plot_settings_A = [
    {"cmap": cm.sdoaia171, "title": "AIA 171", 'vmin': -1, 'vmax': 1},
]
plot_settings_B = [
    {"cmap": cm.sdoaia171, "title": "HRI 174", 'vmin': -1, 'vmax': 1},
]

# setup logging
logging_config = config['logging']
wandb_id = logging_config['wandb_id'] if 'wandb_id' in logging_config else None
log_model = logging_config['wandb_log_model'] if 'wandb_log_model' in logging_config else False
wandb_logger = WandbLogger(project=logging_config['wandb_project'], name=logging_config['wandb_name'], offline=False,
                           entity=logging_config['wandb_entity'], id=wandb_id, dir=config['base_dir'], log_model=log_model)
wandb_logger.experiment.config.update(config, allow_val_change=True)



# Start training
module = ITIModule(**config['model'])

# setup save callbacks
checkpoint_callback = ModelCheckpoint(dirpath=base_dir, save_last=True, every_n_epochs=1, save_weights_only=False)
save_callback = SaveCallback(base_dir)

# setup plot callbacks
#prediction_dir = os.path.join(base_dir, 'prediction')
#os.makedirs(prediction_dir, exist_ok=True)
plot_callbacks = []
plot_callbacks += [PlotBAB(hri_valid.sample(4), module, plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]
plot_callbacks += [PlotABA(aia_valid.sample(4), module, plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]

n_gpus = torch.cuda.device_count()
trainer = Trainer(max_epochs=int(config['training']['epochs']),
                  logger=wandb_logger,
                  devices=n_gpus if n_gpus > 0 else None,
                  accelerator="gpu" if n_gpus >= 1 else None,
                  strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                  num_sanity_val_steps=0,
                  callbacks=[checkpoint_callback, save_callback, *plot_callbacks],)

trainer.fit(module, data_module, ckpt_path='last')