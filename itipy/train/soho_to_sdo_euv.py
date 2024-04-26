import argparse
import logging
import os

from sunpy.visualization.colormaps import cm

from itipy.data.dataset import SDODataset, SOHODataset, StorageDataset
from itipy.data.editor import RandomPatchEditor, LambdaEditor
from itipy.train.model import DiscriminatorMode
from itipy.trainer import Trainer

parser = argparse.ArgumentParser(description='Train SOHO-To-SDO translations without magnetograms')
parser.add_argument('--base_dir', type=str, help='path to the results directory.')

parser.add_argument('--sdo_path', type=str, help='path to the SDO data.')
parser.add_argument('--soho_path', type=str, help='path to the SOHO data.')
parser.add_argument('--sdo_converted_path', type=str, help='path to store the converted SDO data.')
parser.add_argument('--soho_converted_path', type=str, help='path to store the converted SOHO data.')

args = parser.parse_args()
base_dir = args.base_dir

sdo_path = args.sdo_path
soho_path = args.soho_path
sdo_converted_path = args.sdo_converted_path
soho_converted_path = args.soho_converted_path

prediction_dir = os.path.join(base_dir, 'prediction')
os.makedirs(prediction_dir, exist_ok=True)

log_iteration = 1000
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(base_dir, "info_log")),
        logging.StreamHandler()
    ])

# Init Model
trainer = Trainer(4, 4, upsampling=1, discriminator_mode=DiscriminatorMode.CHANNELS,
                  lambda_diversity=0, norm='in_rs_aff')
trainer.cuda()

channel_editor = LambdaEditor(lambda x, **kwargs: x[:-1])

# Init Dataset
train_months = list(range(2, 10))
test_months = [11, 12]

sdo_train = SDODataset(sdo_path, patch_shape=(1024, 1024), resolution=2048, months=train_months,
                       years=list(range(2011, 2021)))
sdo_train = StorageDataset(sdo_train, sdo_converted_path,
                           ext_editors=[channel_editor, RandomPatchEditor((256, 256))])

soho_train = SOHODataset(soho_path, resolution=1024, months=train_months)

soho_train = StorageDataset(soho_train, soho_converted_path,
                            ext_editors=[channel_editor, RandomPatchEditor((128, 128))])

sdo_valid = SDODataset(sdo_path, patch_shape=(1024, 1024), resolution=2048, months=test_months, limit=100)
sdo_valid = StorageDataset(sdo_valid, sdo_converted_path, ext_editors=[channel_editor, RandomPatchEditor((256, 256))])

soho_valid = SOHODataset(soho_path, resolution=1024, months=test_months, limit=100)
soho_valid = StorageDataset(soho_valid, soho_converted_path,
                            ext_editors=[channel_editor, RandomPatchEditor((512, 512))])

plot_settings_A = [
    {"cmap": cm.sdoaia171, "title": "EIT 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia193, "title": "EIT 195", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia211, "title": "EIT 284", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "EIT 304", 'vmin': -1, 'vmax': 1},
]
plot_settings_B = [
    {"cmap": cm.sdoaia171, "title": "AIA 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia193, "title": "AIA 193", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia211, "title": "AIA 211", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "AIA 304", 'vmin': -1, 'vmax': 1},
]

trainer.startBasicTraining(base_dir, soho_train, sdo_train, soho_valid, sdo_valid, plot_settings_A, plot_settings_B)
