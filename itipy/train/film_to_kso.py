import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader

from itipy.callback import PlotBAB, PlotABA, VariationPlotBA, HistoryCallback, ProgressCallback, \
    SaveCallback
from itipy.data.dataset import KSOFlatDataset, StorageDataset, KSOFilmDataset
from itipy.data.editor import RandomPatchEditor
from itipy.trainer import Trainer, loop

parser = argparse.ArgumentParser(description='Train KSO Film-To-CCD translations')
parser.add_argument('--base_dir', type=str, help='path to the results directory.')

parser.add_argument('--kso_path', type=str, help='path to the high-quality KSO data.')
parser.add_argument('--film_path', type=str, help='path to the film KSO data.')
parser.add_argument('--kso_converted_path', type=str, help='path to store the converted KSO data.')
parser.add_argument('--film_converted_path', type=str, help='path to store the converted KSO-film data.')
parser.add_argument('--resolution', type=int, help='resolution of the images (default=512).', default=512,
                    required=False)

args = parser.parse_args()

base_dir = args.base_dir
resolution = args.resolution
kso_path = args.kso_path
film_path = args.film_path
kso_converted_path = args.kso_converted_path
film_converted_path = args.film_converted_path

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
trainer.train()
start_it = trainer.resume(base_dir)

# Init Dataset
ccd_dataset = KSOFlatDataset(kso_path, resolution, months=list(range(12)))
film_dataset = KSOFilmDataset(film_path, resolution, months=list(range(12)))
ccd_storage = StorageDataset(ccd_dataset, kso_converted_path, ext_editors=[RandomPatchEditor((256, 256))])
film_storage = StorageDataset(film_dataset, film_converted_path, ext_editors=[RandomPatchEditor((256, 256))])

ccd_plot = StorageDataset(ccd_dataset, kso_converted_path)
film_plot = StorageDataset(film_dataset, film_converted_path)

kso_ccd_iterator = loop(DataLoader(ccd_storage, batch_size=1, shuffle=True, num_workers=8))
kso_film_iterator = loop(DataLoader(film_storage, batch_size=1, shuffle=True, num_workers=8))

# Init Plot Callbacks
history = HistoryCallback(trainer, base_dir)
progress = ProgressCallback(trainer)
save = SaveCallback(trainer, base_dir)

plot_settings_A = {"cmap": "gray", "title": "Quality A", 'vmin': -1, 'vmax': 1}
plot_settings_B = {"cmap": "gray", "title": "Quality B", 'vmin': -1, 'vmax': 1}

log_iteration = 1000
bab_callback = PlotBAB(ccd_plot.sample(3), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

aba_callback = PlotABA(film_plot.sample(3), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

cutout_callback = PlotABA(film_storage.sample(6), trainer, prediction_dir, log_iteration=log_iteration,
                          plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B, plot_id='CUTOUT')

v_callback = VariationPlotBA(ccd_plot.sample(3), trainer, prediction_dir, 4, log_iteration=log_iteration,
                             plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

aba_callback.call(0)
bab_callback.call(0)
cutout_callback.call(0)
callbacks = [history, progress, save, bab_callback, aba_callback, cutout_callback, v_callback]

# Start training
for it in range(start_it, int(1e8)):
    if it > 100000:
        trainer.gen_ab.eval()  # fix running stats
        trainer.gen_ba.eval()  # fix running stats
    x_a, x_b = next(kso_film_iterator), next(kso_ccd_iterator)
    x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
    #
    trainer.discriminator_update(x_a, x_b)
    trainer.generator_update(x_a, x_b)
    torch.cuda.synchronize()
    #
    for callback in callbacks:
        callback(it)
