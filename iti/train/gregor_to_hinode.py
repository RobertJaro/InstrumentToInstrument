import glob
import logging
import os

from iti.data.editor import RandomPatchEditor, RandomPatch3DEditor
from iti.train.model import DiscriminatorMode

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.utils.data import DataLoader

from iti.data.dataset import StorageDataset, HinodeDataset, GregorDataset
from iti.callback import PlotBAB, PlotABA, VariationPlotBA, HistoryCallback, ProgressCallback, \
    SaveCallback
from iti.trainer import Trainer, loop

base_dir = "/gss/r.jarolim/iti/gregor_hinode_v2"
gregor_path = "/gss/r.jarolim/data/gregor"
hinode_path = '/gss/r.jarolim/data/hinode/gband/*.fits'
gregor_converted_path = '/gss/r.jarolim/data/converted/gregor_train'
hinode_converted_path = '/gss/r.jarolim/data/converted/hinode_gband_train'
n_samples = 6
prediction_dir = os.path.join(base_dir, 'prediction')
os.makedirs(prediction_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(base_dir, "info_log")),
        logging.StreamHandler()
    ])

# Init Model
trainer = Trainer(n_samples, 1, lambda_diversity=0, discriminator_mode=DiscriminatorMode.SINGLE_PER_CHANNEL)
trainer.cuda()
trainer.train()
start_it = trainer.resume(base_dir)

# Find Hinode Files
hinode_files = sorted(glob.glob(hinode_path))

# Init Dataset
gregor_dataset = GregorDataset(gregor_path)
gregor_dataset = StorageDataset(gregor_dataset,
                                gregor_converted_path,
                                ext_editors=[RandomPatch3DEditor((n_samples, 256, 256))])

hinode_dataset = StorageDataset(HinodeDataset(hinode_files, scale=0.056, wavelength='gband'),
                                hinode_converted_path,
                                ext_editors=[RandomPatchEditor((256, 256))])

gregor_iterator = loop(DataLoader(gregor_dataset, batch_size=1, shuffle=True, num_workers=8))
hinode_iterator = loop(DataLoader(hinode_dataset, batch_size=1, shuffle=True, num_workers=8))

# Init Callbacks
history = HistoryCallback(trainer, base_dir)
progress = ProgressCallback(trainer)
save = SaveCallback(trainer, base_dir)

plot_settings_A = {"cmap": "gray", "title": "Gregor G-Band", 'vmin': -1, 'vmax': 1}
plot_settings_B = {"cmap": "gray", "title": "Hinode G-Band", 'vmin': -1, 'vmax': 1}

log_iteration = 1000
bab_callback = PlotBAB(hinode_dataset.sample(4), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

aba_callback = PlotABA(gregor_dataset.sample(4), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

gregor_plot_dataset = GregorDataset(gregor_path)
gregor_plot_dataset = StorageDataset(gregor_plot_dataset,
                                     gregor_converted_path,
                                     ext_editors=[RandomPatch3DEditor((n_samples, 1024, 1024))])
aba_full_callback = PlotABA(gregor_plot_dataset.sample(4), trainer, prediction_dir, log_iteration=log_iteration,
                            plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B, plot_id='ABA_FULL', batch_size=1)

v_callback = VariationPlotBA(hinode_dataset.sample(4), trainer, prediction_dir, 4, log_iteration=log_iteration,
                             plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

callbacks = [history, progress, save, bab_callback, aba_callback, aba_full_callback, v_callback]

aba_callback.call(0)
aba_full_callback.call(0)
bab_callback.call(0)

# Start training
for it in range(start_it, int(1e8)):
    if it > 250000:
        trainer.eval()  # fix running stats
    x_a, x_b = next(gregor_iterator), next(hinode_iterator)
    x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
    #
    trainer.discriminator_update(x_a, x_b)
    trainer.generator_update(x_a, x_b)
    torch.cuda.synchronize()
    #
    for callback in callbacks:
        callback(it)
