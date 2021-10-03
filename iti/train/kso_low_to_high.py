import logging
import os

import torch

from iti.data.editor import BrightestPixelPatchEditor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from torch.utils.data import DataLoader

from iti.data.dataset import StorageDataset, KSOFlatDataset
from iti.callback import PlotBAB, PlotABA, VariationPlotBA, HistoryCallback, ProgressCallback, \
    SaveCallback
from iti.trainer import Trainer, loop

base_dir = "/gss/r.jarolim/iti/kso_quality_1024_v11"
resolution = 1024
low_path = "/gss/r.jarolim/data/anomaly_data_set/quality2"
low_converted_path = '/gss/r.jarolim/data/converted/iti/kso_anomaly_q2_flat_%d' % resolution
high_path = "/gss/r.jarolim/data/kso_synoptic"
high_converted_path = '/gss/r.jarolim/data/converted/iti/kso_synoptic_q1_flat_%d' % resolution

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

q1_dataset = KSOFlatDataset(high_path, resolution, months=list(range(11)))
q1_storage = StorageDataset(q1_dataset, high_converted_path,
                            ext_editors=[BrightestPixelPatchEditor((256, 256), random_selection=0.8)])

q2_dataset = KSOFlatDataset(low_path, resolution, months=list(range(11)))
q2_storage = StorageDataset(q2_dataset, low_converted_path,
                            ext_editors=[BrightestPixelPatchEditor((256, 256), random_selection=0.8)])

q1_full_disk = StorageDataset(q1_dataset, high_converted_path)
q2_full_disk = StorageDataset(q2_dataset, low_converted_path)

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

aba_full_disk = PlotABA(q2_full_disk.sample(2), trainer, prediction_dir, log_iteration=log_iteration,
                        plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B, dpi=300, plot_id='FULL_ABA')

bab_full_disk = PlotBAB(q1_full_disk.sample(2), trainer, prediction_dir, log_iteration=log_iteration,
                        plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B, dpi=300, plot_id='FULL_BAB')

v_callback = VariationPlotBA(q1_storage.sample(8), trainer, prediction_dir, 4, log_iteration=log_iteration,
                             plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)

callbacks = [history, progress, save, bab_callback, aba_callback, v_callback, aba_full_disk, bab_full_disk]

# Start training
for _ in range(100):
    trainer.fill_stack(next(q2_iterator).float().cuda().detach(),
                       next(q1_iterator).float().cuda().detach())
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
