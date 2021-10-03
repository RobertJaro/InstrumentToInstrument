import logging
import os

from iti.data.editor import RandomPatchEditor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.utils.data import DataLoader

from iti.data.dataset import KSOFlatDataset, StorageDataset, KSOFilmDataset
from iti.callback import PlotBAB, PlotABA, VariationPlotBA, HistoryCallback, ProgressCallback, \
    SaveCallback
from iti.trainer import Trainer, loop

base_dir = "/gss/r.jarolim/iti/film_v9"

resolution = 512
kso_path = "/gss/r.jarolim/data/kso_synoptic"
film_path = "/gss/r.jarolim/data/filtered_kso_plate"
kso_converted_path = '/gss/r.jarolim/data/converted/iti/kso_synoptic_q1_flat_%d' % resolution
film_converted_path = '/gss/r.jarolim/data/converted/iti/kso_film_%d' % resolution

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
