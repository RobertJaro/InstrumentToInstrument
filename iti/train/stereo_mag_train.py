import logging
import os

from astropy.visualization import AsinhStretch
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from iti.data.editor import RandomPatchEditor, LambdaEditor, BrightestPixelPatchEditor, BlockReduceEditor
from iti.train.model import DiscriminatorMode

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.utils.data import DataLoader

from iti.data.dataset import SDODataset, StorageDataset, STEREODataset
from iti.evaluation.callback import PlotBAB, PlotABA, HistoryCallback, ProgressCallback, \
    SaveCallback
from iti.train.trainer import Trainer, loop
import numpy as np

base_dir = "/gss/r.jarolim/iti/stereo_mag_v11"
prediction_dir = os.path.join(base_dir, 'prediction')
os.makedirs(prediction_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(base_dir, "info_log")),
        logging.StreamHandler()
    ])

# Init Model
trainer = Trainer(4, 5, upsampling=0, discriminator_mode=DiscriminatorMode.CHANNELS, lambda_diversity=0,
                  norm='in_rs_aff')
trainer.cuda()
start_it = trainer.resume(base_dir)

# Init Dataset
def absolute_mag(data, **kwargs):
    data[-1] = np.abs(data[-1]) * 2 - 1
    return data

abs_mag_editor = LambdaEditor(absolute_mag)
block_reduce_editor = BlockReduceEditor(block_size=(1, 2, 2))

sdo_train = SDODataset("/gss/r.jarolim/data/ch_detection", resolution=2048, patch_shape=(1024, 1024), months=list(range(11)))
sdo_train = StorageDataset(sdo_train, '/gss/r.jarolim/data/converted/sdo_2048',
                           ext_editors=[RandomPatchEditor((512, 512)), block_reduce_editor, abs_mag_editor])

stereo_train = STEREODataset("/gss/r.jarolim/data/stereo_iti2021_prep")
stereo_train = StorageDataset(stereo_train, '/gss/r.jarolim/data/converted/stereo_1024',
                              ext_editors=[BrightestPixelPatchEditor((512, 512)), RandomPatchEditor((256, 256))])

sdo_valid = SDODataset("/gss/r.jarolim/data/ch_detection", resolution=2048)
sdo_valid.addEditor(block_reduce_editor)
sdo_valid.addEditor(abs_mag_editor)
stereo_valid = STEREODataset("/gss/r.jarolim/data/stereo_iti2021_prep")

sdo_iterator = loop(DataLoader(sdo_train, batch_size=1, shuffle=True, num_workers=8))
stereo_iterator = loop(DataLoader(stereo_train, batch_size=1, shuffle=True, num_workers=8))

# Init Plot Callbacks
history = HistoryCallback(trainer, base_dir)
progress = ProgressCallback(trainer)
save = SaveCallback(trainer, base_dir)

plot_settings_A = [
    {"cmap": cm.sdoaia171, "title": "SECCHI 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia193, "title": "SECCHI 195", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia211, "title": "SECCHI 284", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "SECCHI 304", 'vmin': -1, 'vmax': 1}
]
plot_settings_B = [
    {"cmap": cm.sdoaia171, "title": "AIA 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia193, "title": "AIA 193", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia211, "title": "AIA 211", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "AIA 304", 'vmin': -1, 'vmax': 1},
    {"cmap": "gray", "title": "HMI Magnetogram", 'vmin': -1, 'vmax': 1}
]

log_iteration = 1000

cutout_aba_callback = PlotABA(stereo_train.sample(6), trainer, prediction_dir, log_iteration=log_iteration,
                              plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B, plot_id='ABA')
cutout_aba_callback.call(0)

cutout_bab_callback = PlotBAB(sdo_train.sample(6), trainer, prediction_dir, log_iteration=log_iteration,
                              plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B, plot_id='BAB')
cutout_bab_callback.call(0)

aba_callback = PlotABA(stereo_valid.sample(4), trainer, prediction_dir, log_iteration=log_iteration, batch_size=1,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B, plot_id='full_disc_ABA')
aba_callback.call(0)

bab_callback = PlotBAB(sdo_valid.sample(4), trainer, prediction_dir, log_iteration=log_iteration, batch_size=1,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B, plot_id='full_disc_BAB')
bab_callback.call(0)

callbacks = [history, progress, save, aba_callback, bab_callback, cutout_aba_callback, cutout_bab_callback]

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
