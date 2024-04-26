import argparse
import logging
import os
from random import sample

from itipy.data.editor import RandomPatchEditor

from itipy.data.dataset import StorageDataset, HMIContinuumDataset, HinodeDataset
from itipy.trainer import Trainer

import pandas as pd

parser = argparse.ArgumentParser(description='Train HMI-To-Hinode translations')
parser.add_argument('--base_dir', type=str, help='path to the results directory.')

parser.add_argument('--hinode_path', type=str, help='path to the Hinode data.')
parser.add_argument('--hinode_file_list', type=str, help='path to the Hinode file list (see pyiti.data.hinode.classify).')
parser.add_argument('--hmi_path', type=str, help='path to the HMI data.')
parser.add_argument('--hinode_converted_path', type=str, help='path to store the converted Hinode data.')
parser.add_argument('--hmi_converted_path', type=str, help='path to store the converted HMI data.')

args = parser.parse_args()

base_dir = args.base_dir
hmi_path = args.hmi_path
hmi_converted_path = args.hmi_converted_path
hinode_converted_path = args.hinode_converted_path
hinode_file_list = args.hinode_file_list

prediction_dir = os.path.join(base_dir, 'prediction')
os.makedirs(prediction_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(base_dir, "info_log")),
        logging.StreamHandler()
    ])

# Init Model
trainer = Trainer(1, 1, upsampling=2, norm='in_rs_aff', lambda_diversity=0)
trainer.cuda()

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

plot_settings_A = [
    {"cmap": "gray", "title": "HMI Continuum", 'vmin': -1, 'vmax': 1, },
]
plot_settings_B = [
    {"cmap": "gray", "title": "Hinode Continuum", 'vmin': -1, 'vmax': 1},
]

trainer.startBasicTraining(base_dir, hmi_train, hinode_train, hmi_valid, hinode_valid, plot_settings_A, plot_settings_B)
