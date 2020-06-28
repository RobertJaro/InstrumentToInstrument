import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.utils.data import DataLoader

from iti.data.dataset import KSODataset, StorageDataset
from iti.evaluation.callback import PlotBAB, PlotABA, VariationPlotBA, HistoryCallback, ProgressCallback, \
    SaveCallback
from iti.train.trainer import Trainer

base_dir = "/gss/r.jarolim/prediction/iti/kso_quality"
prediction_dir = os.path.join(base_dir, 'prediction')
os.makedirs(prediction_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(base_dir, "info_log")),
        logging.StreamHandler()
    ])

# Init Model
trainer = Trainer(1, 1, 256, 3)
trainer.cuda()
start_it = trainer.resume(base_dir)

# Init Dataset
q1_dataset = StorageDataset(KSODataset("/gss/r.jarolim/data/kso_general/quality1"), '/gss/r.jarolim/data/converted/iti/kso_q1')
q2_dataset = StorageDataset(KSODataset("/gss/r.jarolim/data/kso_general/quality2"), '/gss/r.jarolim/data/converted/iti/kso_q2')

q1_iterator = iter(DataLoader(q1_dataset, batch_size=1, shuffle=True, num_workers=8))
q2_iterator = iter(DataLoader(q2_dataset, batch_size=1, shuffle=True, num_workers=8))

# Init Plot Callbacks
history = HistoryCallback(trainer, base_dir)
progress = ProgressCallback(trainer)
save = SaveCallback(trainer, base_dir)

plot_settings_A = {"cmap": "gray", "title": "Quality 2", 'vmin':-1, 'vmax':1}
plot_settings_B = {"cmap": "gray", "title": "Quality 1", 'vmin':-1, 'vmax':1}

bab_callback = PlotBAB(q1_dataset.sample(8), trainer, prediction_dir, log_iteration=200,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

aba_callback = PlotABA(q2_dataset.sample(8), trainer, prediction_dir, log_iteration=200,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

v_callback = VariationPlotBA(q1_dataset.sample(8), trainer, prediction_dir, 4, log_iteration=200,
                             plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

callbacks = [history, progress, save, bab_callback, aba_callback, v_callback]
for it in range(start_it, int(1e8)):
    x_a, x_b = next(q1_iterator), next(q2_iterator)
    x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
    #
    trainer.discriminator_update(x_a, x_b)
    trainer.generator_update(x_a, x_b)
    torch.cuda.synchronize()
    #
    for callback in callbacks:
        callback(it)
