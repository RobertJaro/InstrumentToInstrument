import argparse
import os

import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sunpy.visualization.colormaps import cm

from iti.callback import SaveCallback, PlotBAB, PlotABA
from iti.data.dataset import SDODataset, StorageDataset, STEREODataset
from iti.data.data_module import ITIDataModule
from iti.data.editor import RandomPatchEditor, SliceEditor, BrightestPixelPatchEditor
from iti.iti import ITIModule

parser = argparse.ArgumentParser(description='Train STEREO-To-SDO translations')
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
stereo_path = data_config['A_path']
stereo_converted_path = data_config['converted_A_path']
sdo_path = data_config['B_path']
sdo_converted_path = data_config['converted_B_path']

test_months = [11, 12]
train_months = list(range(2, 10))

sdo_dataset = SDODataset(sdo_path, resolution=4096, patch_shape=(512, 512), months=train_months)
sdo_dataset = StorageDataset(sdo_dataset,
                             sdo_converted_path,
                             ext_editors=[SliceEditor(0, -1),
                                          RandomPatchEditor((512, 512))])

stereo_dataset = STEREODataset(stereo_path, months=train_months, patch_shape=(512, 512))
stereo_dataset = StorageDataset(stereo_dataset, stereo_converted_path,
                                ext_editors=[BrightestPixelPatchEditor((256, 256)), RandomPatchEditor((128, 128))])

sdo_valid = StorageDataset(SDODataset(sdo_path, resolution=4096, patch_shape=(512, 512), months=test_months, limit=10),
                           sdo_converted_path, ext_editors=[SliceEditor(0, -1)])
stereo_valid = StorageDataset(STEREODataset(stereo_path, patch_shape=(128, 128), months=test_months, limit=10),
                              stereo_converted_path, ext_editors=[BrightestPixelPatchEditor((128, 128))])

data_module = ITIDataModule(stereo_dataset, sdo_dataset, stereo_valid, sdo_valid, **config['data'])

plot_settings_A = [
    {"cmap": cm.sdoaia171, "title": "SECCHI 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia193, "title": "SECCHI 195", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia211, "title": "SECCHI 284", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "SECCHI 304", 'vmin': -1, 'vmax': 1},
]
plot_settings_B = [
    {"cmap": cm.sdoaia171, "title": "AIA 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia193, "title": "AIA 193", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia211, "title": "AIA 211", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "AIA 304", 'vmin': -1, 'vmax': 1},
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
plot_callbacks += [PlotBAB(sdo_valid.sample(4), module, plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]
plot_callbacks += [PlotABA(stereo_valid.sample(4), module, plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]

n_gpus = torch.cuda.device_count()
trainer = Trainer(max_epochs=int(config['training']['epochs']),
                  logger=wandb_logger,
                  devices=n_gpus if n_gpus > 0 else None,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                  num_sanity_val_steps=-1,
                  callbacks=[checkpoint_callback, save_callback, *plot_callbacks], )

trainer.fit(module, data_module, ckpt_path='last')
