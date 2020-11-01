import glob
import logging
import os
from random import sample

from astropy.io.fits import getdata, getheader
from tqdm import tqdm

from iti.data.editor import RandomPatchEditor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.utils.data import DataLoader

from iti.data.dataset import StorageDataset, HMIContinuumDataset, HinodeDataset
from iti.evaluation.callback import PlotBAB, PlotABA, VariationPlotBA, HistoryCallback, ProgressCallback, \
    SaveCallback
from iti.train.trainer import Trainer, loop

import numpy as np

base_dir = "/gss/r.jarolim/iti/hmi_hinode_v12"
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
trainer.train()
start_it = trainer.resume(base_dir)

# Find Hinode Files
hinode_files = sorted(glob.glob('/gss/r.jarolim/data/hinode/level1/*.fits'))
special_features = []
quite_sun = []
for f in tqdm(hinode_files):
    data = np.ravel(getdata(f)) / getheader(f)['EXPTIME']
    if np.sum(data < 25000) > 2000:
        special_features.append(f)
    else:
        quite_sun.append(f)
hinode_files = special_features + sample(quite_sun, len(special_features))

# Init Dataset
hmi_dataset = HMIContinuumDataset("/gss/r.jarolim/data/hmi_continuum/6173", (256, 256))
hmi_dataset = StorageDataset(hmi_dataset,
                             '/gss/r.jarolim/data/converted/hmi_train',
                             ext_editors=[RandomPatchEditor((160, 160))])

hinode_dataset = StorageDataset(HinodeDataset(hinode_files),
                                '/gss/r.jarolim/data/converted/hinode_train',
                                ext_editors=[RandomPatchEditor((640, 640))])

hmi_iterator = loop(DataLoader(hmi_dataset, batch_size=1, shuffle=True, num_workers=8))
hinode_iterator = loop(DataLoader(hinode_dataset, batch_size=1, shuffle=True, num_workers=8))

# Init Callbacks
history = HistoryCallback(trainer, base_dir)
progress = ProgressCallback(trainer)
save = SaveCallback(trainer, base_dir)

plot_settings_A = [
    {"cmap": "gray", "title": "HMI Continuum", 'vmin': -1, 'vmax': 1}
]
plot_settings_B = [
    {"cmap": "gray", "title": "Hinode Continuum", 'vmin': -1, 'vmax': 1},
]

log_iteration = 1000
bab_callback = PlotBAB(hinode_dataset.sample(4), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

aba_callback = PlotABA(hmi_dataset.sample(4), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

v_callback = VariationPlotBA(hinode_dataset.sample(4), trainer, prediction_dir, 4, log_iteration=log_iteration,
                             plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

callbacks = [history, progress, save, bab_callback, aba_callback, v_callback]

# Init generator stack
# trainer.fill_stack([(next(soho_iterator).float().cuda().detach(),
#                      next(sdo_iterator).float().cuda().detach()) for _ in range(50)])
# Start training
for it in range(start_it, int(1e8)):
    if it > 100000:
        trainer.eval()  # fix running stats
    x_a, x_b = next(hmi_iterator), next(hinode_iterator)
    x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
    #
    trainer.discriminator_update(x_a, x_b)
    trainer.generator_update(x_a, x_b)
    torch.cuda.synchronize()
    #
    for callback in callbacks:
        callback(it)
