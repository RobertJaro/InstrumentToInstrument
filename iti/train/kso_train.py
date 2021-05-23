import logging
import os

import torch

from iti.data.editor import RandomPatchEditor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from torch.utils.data import DataLoader

from iti.data.dataset import StorageDataset, KSOFlatDataset
from iti.evaluation.callback import PlotBAB, PlotABA, VariationPlotBA, HistoryCallback, ProgressCallback, \
    SaveCallback
from iti.train.trainer import Trainer, loop

base_dir = "/gss/r.jarolim/iti/kso_quality_1024_v5"
resolution = 1024
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
start_it = trainer.resume(base_dir)

# Init Dataset
q1_dataset = KSOFlatDataset("/gss/r.jarolim/data/kso_synoptic", resolution)
q1_storage = StorageDataset(q1_dataset,
                            '/gss/r.jarolim/data/converted/iti/kso_synoptic_q1_flat_%d' % resolution,
                            ext_editors=[RandomPatchEditor((256, 256))])

q2_dataset = KSOFlatDataset("/gss/r.jarolim/data/kso_general/quality2", resolution)
q2_storage = StorageDataset(q2_dataset,
                            '/gss/r.jarolim/data/converted/iti/kso_q2_flat_%d' % resolution,
                            ext_editors=[RandomPatchEditor((256, 256))])

q1_fulldisc = StorageDataset(q1_dataset, '/gss/r.jarolim/data/converted/iti/kso_synoptic_q1_flat_%d' % resolution)
q2_fulldisc = StorageDataset(q2_dataset, '/gss/r.jarolim/data/converted/iti/kso_q2_flat_%d' % resolution)

q1_iterator = loop(DataLoader(q1_storage, batch_size=1, shuffle=True, num_workers=8))
q2_iterator = loop(DataLoader(q2_storage, batch_size=1, shuffle=True, num_workers=8))

# Init Callbacks
history = HistoryCallback(trainer, base_dir)
progress = ProgressCallback(trainer)
save = SaveCallback(trainer, base_dir)

plot_settings_A = {"cmap": "gray", "title": "Quality 2", 'vmin': -1, 'vmax': 1}
plot_settings_B = {"cmap": "gray", "title": "Quality 1", 'vmin': -1, 'vmax': 1}

log_iteration = 1000
bab_callback = PlotBAB(q1_storage.sample(8), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

aba_callback = PlotABA(q2_storage.sample(8), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B, dpi=300)

aba_fulldisc = PlotABA(q2_fulldisc.sample(2), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B, dpi=300, plot_id='FULL_ABA')

bab_fulldisc = PlotBAB(q1_fulldisc.sample(2), trainer, prediction_dir, log_iteration=log_iteration,
                       plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B, dpi=300, plot_id='FULL_BAB')

v_callback = VariationPlotBA(q1_storage.sample(8), trainer, prediction_dir, 4, log_iteration=log_iteration,
                             plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

callbacks = [history, progress, save, bab_callback, aba_callback, v_callback, aba_fulldisc, bab_fulldisc]

# Start training
for it in range(start_it, int(1e8)):
    if it > 100000:
        trainer.gen_ab.eval()  # fix running stats
        trainer.gen_ba.eval()  # fix running stats
    x_a, x_b = next(q2_iterator), next(q1_iterator)
    x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
    trainer.discriminator_update(x_a, x_b)

    x_a, x_b = next(q2_iterator), next(q1_iterator)
    x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
    trainer.generator_update(x_a, x_b)
    torch.cuda.synchronize()
    #
    for callback in callbacks:
        callback(it)
