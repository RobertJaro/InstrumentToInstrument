import argparse
import logging
import os

from sunpy.visualization.colormaps import cm

from itipy.data.dataset import SDODataset, StorageDataset
from itipy.data.editor import SliceEditor
from itipy.train.model import DiscriminatorMode
from itipy.trainer import Trainer

parser = argparse.ArgumentParser(description='Train SDO EUV-To-magnetogram translations (signed)')
parser.add_argument('--base_dir', type=str, help='path to the results directory.')

parser.add_argument('--sdo_path', type=str, help='path to the SDO data.')
parser.add_argument('--sdo_converted_path', type=str, help='path to store the converted SDO data.')

args = parser.parse_args()
base_dir = args.base_dir

sdo_path = args.sdo_path
sdo_converted_path = args.sdo_converted_path
prediction_dir = os.path.join(base_dir, 'prediction')
os.makedirs(prediction_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(base_dir, "info_log")),
        logging.StreamHandler()
    ])

# Init Model
trainer = Trainer(4, 5, upsampling=0, discriminator_mode=DiscriminatorMode.CHANNELS, lambda_diversity=0,
                  norm='in_aff', n_filters=32)
trainer.cuda()

sdo_train = SDODataset(sdo_path, resolution=512, months=list(range(2, 10)), years=list(range(2011, 2021)))
sdo_euv_train = StorageDataset(sdo_train, sdo_converted_path,
                               ext_editors=[SliceEditor(0, 4)])
sdo_mag_train = StorageDataset(sdo_train, sdo_converted_path, ext_editors=[])

sdo_valid = SDODataset(sdo_path, resolution=512, limit=100, months=[11, 12])
sdo_euv_valid = StorageDataset(sdo_valid, sdo_converted_path,
                               ext_editors=[SliceEditor(0, 4), ])
sdo_mag_valid = StorageDataset(sdo_valid, sdo_converted_path, ext_editors=[])

plot_settings_A = [
    {"cmap": cm.sdoaia171, "title": "SECCHI 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia193, "title": "SECCHI 195", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia211, "title": "SECCHI 284", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "SECCHI 304", 'vmin': -1, 'vmax': 1}
]
plot_settings_B = [
    {"cmap": cm.sdoaia171, "title": "AIA 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia193, "title": "AIA 193", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia211, "title": "AIA 211", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "AIA 304", 'vmin': -1, 'vmax': 1},
    {"cmap": "gray", "title": "HMI Magnetogram", 'vmin': -1, 'vmax': 1}
]

trainer.startBasicTraining(base_dir, sdo_euv_train, sdo_mag_train, sdo_euv_valid, sdo_mag_valid,
                           plot_settings_A, plot_settings_B, num_workers=4)
