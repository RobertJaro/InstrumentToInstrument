import logging
import os

from sunpy.visualization.colormaps import cm

from iti.data.editor import RandomPatchEditor, LambdaEditor
from iti.train.model import DiscriminatorMode

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.utils.data import DataLoader

from iti.data.dataset import SDODataset, SOHODataset, StorageDataset
from iti.callback import PlotBAB, PlotABA, VariationPlotBA, HistoryCallback, ProgressCallback, \
    SaveCallback, ValidationHistoryCallback
from iti.trainer import Trainer, loop

base_dir = "/gss/r.jarolim/iti/soho_sdo_euv_v1"

sdo_path = "/gss/r.jarolim/data/ch_detection"
soho_path = "/gss/r.jarolim/data/soho_iti2021_prep"
sdo_converted_path = '/gss/r.jarolim/data/converted/sdo_2048'
soho_converted_path = '/gss/r.jarolim/data/converted/soho_1024'

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
trainer.train()
start_it = trainer.resume(base_dir)

channel_editor = LambdaEditor(lambda x, **kwargs: x[:-1])

# Init Dataset
sdo_train = SDODataset(sdo_path, patch_shape=(1024, 1024), resolution=2048, months=list(range(11)),
                       years=list(range(2011, 2021)))
sdo_train = StorageDataset(sdo_train, sdo_converted_path,
                           ext_editors=[channel_editor, RandomPatchEditor((256, 256))])

soho_train = SOHODataset(soho_path, resolution=1024, months=list(range(11)))

soho_train = StorageDataset(soho_train, soho_converted_path,
                            ext_editors=[channel_editor, RandomPatchEditor((128, 128))])

sdo_valid = SDODataset(sdo_path, patch_shape=(1024, 1024), resolution=2048, months=[11, 12], limit=100)
sdo_valid = StorageDataset(sdo_valid, sdo_converted_path, ext_editors=[channel_editor, RandomPatchEditor((256, 256))])

soho_valid = SOHODataset(soho_path, resolution=1024, months=[11, 12], limit=100)
soho_valid = StorageDataset(soho_valid, soho_converted_path,
                            ext_editors=[channel_editor, RandomPatchEditor((128, 128))])

sdo_iterator = loop(DataLoader(sdo_train, batch_size=1, shuffle=True, num_workers=8))
soho_iterator = loop(DataLoader(soho_train, batch_size=1, shuffle=True, num_workers=8))

# Init Callbacks
history = HistoryCallback(trainer, base_dir)
validation = ValidationHistoryCallback(trainer, soho_valid, sdo_valid, base_dir, log_iteration)
progress = ProgressCallback(trainer)
save = SaveCallback(trainer, base_dir)

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

soho_plot = SOHODataset(soho_path, patch_shape=(1024, 1024), months=[11, 12])
soho_plot.addEditor(channel_editor)
sdo_plot = SDODataset(sdo_path, patch_shape=(2048, 2048), months=[11, 12])
sdo_plot.addEditor(channel_editor)

bab_callback = PlotBAB(sdo_valid.sample(4), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

aba_callback = PlotABA(soho_valid.sample(4), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

full_disc_aba_callback = PlotABA(soho_plot.sample(4),
                                 trainer, prediction_dir, log_iteration=log_iteration, batch_size=1,
                                 plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B,
                                 plot_id='FULL_ABA')

full_disc_bab_callback = PlotBAB(sdo_plot.sample(4),
                                 trainer, prediction_dir, log_iteration=log_iteration, batch_size=1,
                                 plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B,
                                 plot_id='FULL_BAB')

v_callback = VariationPlotBA(sdo_valid.sample(4), trainer, prediction_dir, 4, log_iteration=log_iteration,
                             plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

callbacks = [save, history, progress, bab_callback, aba_callback, v_callback, full_disc_aba_callback,
             full_disc_bab_callback, validation]

# Start training
for it in range(start_it, int(1e8)):
    if it > 100000:
        trainer.gen_ab.eval()  # fix running stats
        trainer.gen_ba.eval()  # fix running stats
    #
    x_a, x_b = next(soho_iterator), next(sdo_iterator)
    x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
    trainer.discriminator_update(x_a, x_b)
    #
    x_a, x_b = next(soho_iterator), next(sdo_iterator)
    x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
    trainer.generator_update(x_a, x_b)
    torch.cuda.synchronize()
    #
    for callback in callbacks:
        callback(it)
