import logging
import os

from sunpy.visualization.colormaps import cm

from iti.data.editor import RandomPatchEditor
from iti.train.model import DiscriminatorMode

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.utils.data import DataLoader

from iti.data.dataset import SDODataset, SOHODataset, StorageDataset
from iti.evaluation.callback import PlotBAB, PlotABA, VariationPlotBA, HistoryCallback, ProgressCallback, \
    SaveCallback, NormScheduler, ValidationHistoryCallback
from iti.train.trainer import Trainer, loop

base_dir = "/gss/r.jarolim/iti/soho_sdo_v23"
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
trainer = Trainer(5, 5, upsampling=1, discriminator_mode=DiscriminatorMode.CHANNELS,
                  lambda_diversity=0, norm='in_rs_aff')
trainer.cuda()
trainer.train()
start_it = trainer.resume(base_dir)

# Init Dataset
sdo_dataset = SDODataset("/gss/r.jarolim/data/ch_detection", patch_shape=(1024, 1024))
sdo_dataset = StorageDataset(sdo_dataset,
                             '/gss/r.jarolim/data/converted/sdo_train',
                             ext_editors=[RandomPatchEditor((256, 256))])

soho_dataset = StorageDataset(SOHODataset("/gss/r.jarolim/data/soho/train"),
                              '/gss/r.jarolim/data/converted/soho_train',
                              ext_editors=[RandomPatchEditor((128, 128))])

sdo_valid = SDODataset("/gss/r.jarolim/data/sdo/valid", patch_shape=(256, 256))
soho_valid = SOHODataset("/gss/r.jarolim/data/soho/valid", patch_shape=(128, 128))

sdo_iterator = loop(DataLoader(sdo_dataset, batch_size=1, shuffle=True, num_workers=8))
soho_iterator = loop(DataLoader(soho_dataset, batch_size=1, shuffle=True, num_workers=8))

# Init Plot Callbacks
history = HistoryCallback(trainer, base_dir)
validation = ValidationHistoryCallback(trainer,
                                       StorageDataset(soho_valid, '/gss/r.jarolim/data/converted/soho_valid'),
                                       StorageDataset(sdo_valid, '/gss/r.jarolim/data/converted/sdo_valid'),
                                       base_dir, log_iteration)
progress = ProgressCallback(trainer)
save = SaveCallback(trainer, base_dir)

plot_settings_A = [
    {"cmap": cm.sdoaia171, "title": "EIT 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia193, "title": "EIT 195", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia211, "title": "EIT 284", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "EIT 304", 'vmin': -1, 'vmax': 1},
    {"cmap": "gray", "title": "MDI Magnetogram", 'vmin': -1, 'vmax': 1}
]
plot_settings_B = [
    {"cmap": cm.sdoaia171, "title": "AIA 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia193, "title": "AIA 193", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia211, "title": "AIA 211", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "AIA 304", 'vmin': -1, 'vmax': 1},
    {"cmap": "gray", "title": "HMI Magnetogram", 'vmin': -1, 'vmax': 1},
]

bab_callback = PlotBAB(sdo_valid.sample(4), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

aba_callback = PlotABA(soho_valid.sample(4), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

full_disc_aba_callback = PlotABA(SOHODataset("/gss/r.jarolim/data/soho/valid", patch_shape=(1024, 1024)).sample(1),
                                 trainer,
                                 prediction_dir, log_iteration=log_iteration,
                                 plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B,
                                 plot_id='full_disc_aba')

full_disc_bab_callback = PlotBAB(SDODataset("/gss/r.jarolim/data/sdo/valid", patch_shape=(2048, 2048)).sample(1),
                                 trainer,
                                 prediction_dir, log_iteration=log_iteration,
                                 plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B,
                                 plot_id='full_disc_bab')

v_callback = VariationPlotBA(sdo_valid.sample(4), trainer, prediction_dir, 4, log_iteration=log_iteration,
                             plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

callbacks = [save, history, progress, bab_callback, aba_callback, v_callback, full_disc_aba_callback,
             full_disc_bab_callback]

# Start training
for it in range(start_it, int(1e8)):
    if it > 250000:
        trainer.eval()  # fix running stats
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
