import logging
import os

from sunpy.visualization.colormaps import cm

from iti.data.editor import RandomPatchEditor, SliceEditor
from iti.train.model import DiscriminatorMode

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.utils.data import DataLoader

from iti.data.dataset import SDODataset, StorageDataset, STEREODataset
from iti.evaluation.callback import PlotBAB, PlotABA, HistoryCallback, ProgressCallback, SaveCallback
from iti.train.trainer import Trainer, loop

base_dir = "/gss/r.jarolim/iti/stereo_v6"
prediction_dir = os.path.join(base_dir, 'prediction')
os.makedirs(prediction_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(base_dir, "info_log")),
        logging.StreamHandler()
    ])

# Init Model
trainer = Trainer(4, 4, upsampling=2, discriminator_mode=DiscriminatorMode.CHANNELS, lambda_diversity=0,
                  norm='in_rs_aff')
trainer.cuda()
start_it = trainer.resume(base_dir)

# Init Dataset
sdo_dataset = SDODataset("/gss/r.jarolim/data/ch_detection", resolution=4096, patch_shape=(1024, 1024))
sdo_dataset = StorageDataset(sdo_dataset,
                             '/gss/r.jarolim/data/converted/sdo_4096_train',
                             ext_editors=[SliceEditor(0, -1),
                                          RandomPatchEditor((512, 512))])

stereo_dataset = StorageDataset(STEREODataset("/gss/r.jarolim/data/stereo_prep/train"),
                                '/gss/r.jarolim/data/converted/stereo_train',
                                ext_editors=[RandomPatchEditor((128, 128))])

sdo_valid = StorageDataset(SDODataset("/gss/r.jarolim/data/sdo/valid", resolution=4096, patch_shape=(1024, 1024)),
                           '/gss/r.jarolim/data/converted/sdo_4096_valid', ext_editors=[RandomPatchEditor((512, 512)), SliceEditor(0, -1)])
stereo_valid = StorageDataset(STEREODataset("/gss/r.jarolim/data/stereo_prep/valid", patch_shape=(1024, 1024)),
                              '/gss/r.jarolim/data/converted/stereo_valid', ext_editors=[RandomPatchEditor((128, 128))])

sdo_iterator = loop(DataLoader(sdo_dataset, batch_size=1, shuffle=True, num_workers=8))
stereo_iterator = loop(DataLoader(stereo_dataset, batch_size=1, shuffle=True, num_workers=8))

# Init Plot Callbacks
history = HistoryCallback(trainer, base_dir)
progress = ProgressCallback(trainer)
save = SaveCallback(trainer, base_dir)

plot_settings_A = [
    {"cmap": cm.sdoaia171, "title": "SECCHI 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia193, "title": "SECCHI 195", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia211, "title": "SECCHI 284", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "SECCHI 304", 'vmin': -1, 'vmax': 1},
]
plot_settings_B = [
    {"cmap": cm.sdoaia171, "title": "AIA 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia193, "title": "AIA 193", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia211, "title": "AIA 211", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "AIA 304", 'vmin': -1, 'vmax': 1},
]

log_iteration = 1000

aba_callback = PlotABA(stereo_valid.sample(4), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

bab_callback = PlotBAB(sdo_valid.sample(4), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

full_disc_aba_callback = PlotABA(STEREODataset("/gss/r.jarolim/data/stereo_prep/valid").sample(4),
                                 trainer, prediction_dir, log_iteration=log_iteration, batch_size=1,
                                 plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B,
                                 plot_id='full_disc_aba')

callbacks = [history, progress, save, aba_callback, bab_callback, full_disc_aba_callback]

# Start training
for it in range(start_it, int(1e8)):
    if it > 100000:
        trainer.gen_ab.eval()  # fix running stats
        trainer.gen_ba.eval()  # fix running stats
    x_a, x_b = next(stereo_iterator), next(sdo_iterator)
    x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
    #
    trainer.discriminator_update(x_a, x_b)
    trainer.generator_update(x_a, x_b)
    torch.cuda.synchronize()
    #
    for callback in callbacks:
        callback(it)
