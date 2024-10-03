import argparse
import logging
import os

from itipy.callback import HistoryCallback, ProgressCallback, \
    SaveCallback
from itipy.data.dataset import StorageDataset, KSOFlatDataset
from itipy.data.editor import BrightestPixelPatchEditor
from itipy.trainer import Trainer

parser = argparse.ArgumentParser(description='Train mitigation of atmospheric degradations from KSO Halpha observations')
parser.add_argument('--base_dir', type=str, help='path to the results directory.')

parser.add_argument('--hq_path', type=str, help='path to the high-quality KSO data.')
parser.add_argument('--lq_path', type=str, help='path to the low-quality KSO data.')
parser.add_argument('--hq_converted_path', type=str, help='path to store the converted high-quality KSO data.')
parser.add_argument('--lq_converted_path', type=str, help='path to store the converted low-quality KSO data.')
parser.add_argument('--resolution', type=int, help='resolution of the images (default=1024).',
                    default=1024, required=False)

args = parser.parse_args()
base_dir = args.base_dir
resolution = args.resolution
low_path = args.lq_path
low_converted_path = args.lq_converted_path
high_path = args.hq_path
high_converted_path = args.hq_converted_path

prediction_dir = os.path.join(base_dir, 'prediction')
os.makedirs(prediction_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(base_dir, "info_log")),
        logging.StreamHandler()
    ])

# Init Model
trainer = Trainer(1, 1)
trainer.cuda()
start_it = trainer.resume(base_dir)

# Init Dataset

train_months = list(range(2, 10))
valid_months = [11, 12]

q1_dataset = KSOFlatDataset(high_path, resolution, months=train_months)
q1_storage = StorageDataset(q1_dataset, high_converted_path,
                            ext_editors=[BrightestPixelPatchEditor((256, 256), random_selection=0.8)])

q2_dataset = KSOFlatDataset(low_path, resolution, months=train_months)
q2_storage = StorageDataset(q2_dataset, low_converted_path,
                            ext_editors=[BrightestPixelPatchEditor((256, 256), random_selection=0.8)])

q1_valid_dataset = KSOFlatDataset(high_path, resolution, months=valid_months)
q1_valid_dataset = StorageDataset(q1_valid_dataset, high_converted_path)

q2_valid_dataset = KSOFlatDataset(low_path, resolution, months=valid_months)
q2_valid_dataset = StorageDataset(q2_valid_dataset, low_converted_path)

# Init Callbacks
history = HistoryCallback(trainer, base_dir)
progress = ProgressCallback(trainer)
save = SaveCallback(trainer, base_dir)

plot_settings_A = {"cmap": "gray", "title": "Quality 2", 'vmin': -1, 'vmax': 1}
plot_settings_B = {"cmap": "gray", "title": "Quality 1", 'vmin': -1, 'vmax': 1}

trainer.startBasicTraining(base_dir, q2_storage, q1_storage, q2_valid_dataset, q1_valid_dataset,
                           plot_settings_A, plot_settings_B, num_workers=8)
