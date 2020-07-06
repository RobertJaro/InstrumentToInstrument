import logging
import os
import time
from random import sample

from sunpy.visualization.colormaps import cm

from iti.data.editor import RandomPatchEditor
from iti.train.model import DiscriminatorMode

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.utils.data import DataLoader

from iti.data.dataset import SDODataset, SOHODataset, StorageDataset
from iti.evaluation.callback import PlotBAB, PlotABA, VariationPlotBA, HistoryCallback, ProgressCallback, \
    SaveCallback, LRScheduler
from iti.train.trainer import Trainer, loop

base_dir = "/gss/r.jarolim/prediction/iti/soho_sdo_v1"
prediction_dir = os.path.join(base_dir, 'prediction')
os.makedirs(prediction_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(base_dir, "info_log")),
        logging.StreamHandler()
    ])

# Init Model
trainer = Trainer(5, 5, depth_generator=2, depth_noise=3, n_filters=64,res_blocks=4, upsampling=1, discriminator_mode=DiscriminatorMode.PER_CHANNEL,
                  lambda_diversity=0, )
trainer.cuda()
start_it = trainer.resume(base_dir)

# Init Dataset
sdo_dataset = SDODataset("/gss/r.jarolim/data/sdo/train", patch_shape=(1024, 1024))
sdo_dataset.data = sample(sdo_dataset.data, 100)
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
progress = ProgressCallback(trainer)
save = SaveCallback(trainer, base_dir)

plot_settings_A = [
    {"cmap": cm.sohoeit171, "title": "EIT 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sohoeit195, "title": "EIT 195", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sohoeit284, "title": "EIT 284", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sohoeit304, "title": "EIT 304", 'vmin': -1, 'vmax': 1},
    {"cmap": "gray", "title": "MDI Magnetogram", 'vmin': -1, 'vmax': 1}
]
plot_settings_B = [
    {"cmap": cm.sdoaia171, "title": "AIA 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia193, "title": "AIA 193", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia211, "title": "AIA 211", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "AIA 304", 'vmin': -1, 'vmax': 1},
    {"cmap": "gray", "title": "HMI Magnetogram", 'vmin': -1, 'vmax': 1},
]

log_iteration = 1000
bab_callback = PlotBAB(sdo_valid.sample(4), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

aba_callback = PlotABA(soho_valid.sample(4), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

full_disc_callback = PlotABA(SOHODataset("/gss/r.jarolim/data/soho/valid", patch_shape=(1024, 1024)).sample(2), trainer,
                             prediction_dir, log_iteration=log_iteration,
                             plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B, plot_id='full_disc')
full_disc_callback.call(0)

small_fov_callback = PlotABA(SOHODataset("/gss/r.jarolim/data/soho/valid", patch_shape=(32, 32)).sample(8), trainer,
                             prediction_dir, log_iteration=log_iteration,
                             plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B, plot_id='small_fov')

v_callback = VariationPlotBA(sdo_valid.sample(4), trainer, prediction_dir, 4, log_iteration=log_iteration,
                             plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

lr_scheduler = LRScheduler(trainer, 30000)

callbacks = [history, progress, save, bab_callback, aba_callback, v_callback, lr_scheduler, full_disc_callback, small_fov_callback]

# Init generator stack
# trainer.fill_stack([(next(soho_iterator).float().cuda().detach(),
#                      next(sdo_iterator).float().cuda().detach()) for _ in range(50)])
# Start training
for it in range(start_it, int(1e8)):
    x_a, x_b = next(soho_iterator), next(sdo_iterator)
    x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
    end = time.time()
    #
    trainer.discriminator_update(x_a, x_b)
    trainer.generator_update(x_a, x_b)
    torch.cuda.synchronize()
    #
    for callback in callbacks:
        callback(it)
