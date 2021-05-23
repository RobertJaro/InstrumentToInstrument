import glob
import logging
import os
from random import sample

from astropy.io.fits import getdata, getheader
from dateutil.parser import parse
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

import pandas as pd

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

df = pd.read_csv('/gss/r.jarolim/data/hinode/file_list.csv', index_col=False, parse_dates=['date'])
train_df = df[(df.date.dt.month != 12) & (df.date.dt.month != 11)]
features = train_df[train_df.classification == 'feature']
quiet = train_df[train_df.classification == 'quiet']
limb = train_df[train_df.classification == 'limb']
hinode_files = list(features.file) + list(limb.file) + sample(list(quiet.file), len(features) + len(limb))

# Init Dataset
hmi_files = glob.glob("/gss/r.jarolim/data/hmi_continuum/6173/*.fits")
hmi_files = [f for f in hmi_files if parse(os.path.basename(f).replace('.fits', '')).month != 12]
hmi_dataset = HMIContinuumDataset(hmi_files, (512, 512))
hmi_dataset = StorageDataset(hmi_dataset, '/gss/r.jarolim/data/converted/hmi_continuum', ext_editors=[RandomPatchEditor((160, 160))])

hinode_dataset = StorageDataset(HinodeDataset(hinode_files), '/gss/r.jarolim/data/converted/hinode_continuum',
                                ext_editors=[RandomPatchEditor((640, 640))])

hmi_iterator = loop(DataLoader(hmi_dataset, batch_size=1, shuffle=True, num_workers=8))
hinode_iterator = loop(DataLoader(hinode_dataset, batch_size=1, shuffle=True, num_workers=8))

logging.info("Using {} HMI samples".format(len(hmi_dataset)))
logging.info("Using {} Hinode samples".format(len(hinode_dataset)))

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
    if it > 250000:
        trainer.gen_ab.eval()  # fix running stats
        trainer.gen_ba.eval()  # fix running stats
    x_a, x_b = next(hmi_iterator), next(hinode_iterator)
    x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
    #
    trainer.discriminator_update(x_a, x_b)
    trainer.generator_update(x_a, x_b)
    torch.cuda.synchronize()
    #
    for callback in callbacks:
        callback(it)
