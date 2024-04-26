import argparse
import logging
import os

import numpy as np
from sunpy.visualization.colormaps import cm

from itipy.data.dataset import SDODataset, StorageDataset, STEREODataset
from itipy.data.editor import RandomPatchEditor, LambdaEditor, BrightestPixelPatchEditor, BlockReduceEditor
from itipy.train.model import DiscriminatorMode
from itipy.trainer import Trainer

parser = argparse.ArgumentParser(description='Train STEREO-To-SDO translations with synthetic magnetograms')
parser.add_argument('--base_dir', type=str, help='path to the results directory.')

parser.add_argument('--sdo_path', type=str, help='path to the SDO data.')
parser.add_argument('--stereo_path', type=str, help='path to the STEREO data.')
parser.add_argument('--sdo_converted_path', type=str, help='path to store the converted SDO data.')
parser.add_argument('--stereo_converted_path', type=str, help='path to store the converted STEREO data.')

args = parser.parse_args()
base_dir = args.base_dir

stereo_path = args.stereo_path
stereo_converted_path = args.stereo_converted_path
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
                  norm='in_rs_aff')
trainer.cuda()
start_it = trainer.resume(base_dir)


# Init Dataset
def absolute_mag(data, **kwargs):
    data[-1] = np.abs(data[-1]) * 2 - 1
    return data


abs_mag_editor = LambdaEditor(absolute_mag)
block_reduce_editor = BlockReduceEditor(block_size=(1, 2, 2))

test_months = [11, 12]
train_months = list(range(2, 10))

sdo_train = SDODataset(sdo_path, resolution=2048, patch_shape=(1024, 1024), months=train_months)
sdo_train = StorageDataset(sdo_train, sdo_converted_path,
                           ext_editors=[RandomPatchEditor((512, 512)), block_reduce_editor, abs_mag_editor])

stereo_train = STEREODataset(stereo_path, months=test_months)
stereo_train = StorageDataset(stereo_train, stereo_converted_path,
                              ext_editors=[BrightestPixelPatchEditor((512, 512)), RandomPatchEditor((256, 256))])

sdo_valid = SDODataset(sdo_path, resolution=2048)
sdo_valid.addEditor(block_reduce_editor)
sdo_valid.addEditor(abs_mag_editor)
stereo_valid = STEREODataset(stereo_path)

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

# Start training
trainer.startBasicTraining(base_dir, stereo_train, sdo_train, stereo_valid, sdo_valid,
                           plot_settings_A, plot_settings_B)
