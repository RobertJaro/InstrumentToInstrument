import gc
import os

import torch
from dateutil.parser import parse
from torch.utils.data import DataLoader
from tqdm import tqdm

from iti.data.dataset import STEREODataset, SOHODataset, SOHOHMIDataset, SDODataset
from iti.data.editor import sdo_norms, soho_norms, stereo_norms

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from iti.prediction.translate import SOHOToSDO, STEREOToSDO

from matplotlib import pyplot as plt

import numpy as np

# init
base_path = '/gss/r.jarolim/iti/euv_comparison'
os.makedirs(base_path, exist_ok=True)
# create translator
translator_soho = SOHOToSDO(model_path='/gss/r.jarolim/iti/soho_sdo_v25/generator_AB.pt')
translator_stereo = STEREOToSDO(model_path='/gss/r.jarolim/iti/stereo_v7/generator_AB.pt')

print('########## load SOHO ##########')
soho_dataset = SOHODataset("/gss/r.jarolim/data/soho_iti2021_prep", resolution=1024, n_samples=100)
soho_iterator = DataLoader(soho_dataset, batch_size=1, shuffle=False, num_workers=12)

soho_dates = [parse(b.split('.')[0]) for b in soho_dataset.basenames]
soho_lc = []
soho_iti_lc = []
with torch.no_grad():
    for soho_img in tqdm(soho_iterator, total=len(soho_iterator)):
        soho_img = soho_img.cuda()
        iti_img = translator_soho.generator(soho_img)
        # flatten batch + remove magnetogram
        iti_img = iti_img[0, :-1].detach().cpu().numpy()
        soho_img = soho_img[0, :-1].detach().cpu().numpy()
        soho_lc += [[np.mean(norm.inverse((img + 1) / 2))
                     for img, norm in
                     zip(soho_img, [soho_norms[171], soho_norms[195], soho_norms[284], soho_norms[304]])]]
        soho_iti_lc += [[np.mean(norm.inverse((img + 1) / 2))
                         for img, norm in
                         zip(iti_img, [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304]])]]

print('########## load additional SOHO ##########')
soho_hmi_dataset = SOHOHMIDataset('/gss/r.jarolim/data/soho_iti2021_prep', '/gss/r.jarolim/data/ch_detection/6173',
                                  resolution=1024, n_samples=100)
soho_hmi_iterator = DataLoader(soho_hmi_dataset, batch_size=1, shuffle=False, num_workers=12)

soho_dates += [parse(b.split('.')[0]) for b in soho_hmi_dataset.basenames]
with torch.no_grad():
    for soho_img in tqdm(soho_hmi_iterator, total=len(soho_hmi_iterator)):
        soho_img = soho_img.cuda()
        iti_img = translator_soho.generator(soho_img)
        # flatten batch + remove magnetogram
        iti_img = iti_img[0, :-1].detach().cpu().numpy()
        soho_img = soho_img[0, :-1].detach().cpu().numpy()
        soho_lc += [[np.mean(norm.inverse((img + 1) / 2))
                     for img, norm in
                     zip(soho_img, [soho_norms[171], soho_norms[195], soho_norms[284], soho_norms[304]])]]
        soho_iti_lc += [[np.mean(norm.inverse((img + 1) / 2))
                         for img, norm in
                         zip(iti_img, [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304]])]]

# sort by date
soho_dates = np.array(soho_dates)
soho_lc = np.array(soho_lc)
soho_iti_lc = np.array(soho_iti_lc)
idx = np.argsort(soho_dates)
soho_dates = soho_dates[idx]
soho_lc = soho_lc[idx]
soho_iti_lc = soho_iti_lc[idx]

print('########## load STEREO ##########')
stereo_dataset = STEREODataset("/gss/r.jarolim/data/stereo_iti2021_prep", n_samples=100)
stereo_iterator = DataLoader(stereo_dataset, batch_size=1, shuffle=False, num_workers=12)

stereo_dates = [parse(b.split('.')[0]) for b in stereo_dataset.basenames]
stereo_lc = []
stereo_iti_lc = []
with torch.no_grad():
    for stereo_img in tqdm(stereo_iterator, total=len(stereo_iterator)):
        stereo_img = stereo_img.cuda()
        iti_img = translator_stereo.generator(stereo_img)
        iti_img = iti_img[0].detach().cpu().numpy()
        stereo_img = stereo_img[0].detach().cpu().numpy()
        stereo_lc += [[np.mean(norm.inverse((img + 1) / 2))
                       for img, norm in
                       zip(stereo_img, [stereo_norms[171], stereo_norms[195], stereo_norms[284], stereo_norms[304]])]]
        stereo_iti_lc += [[np.mean(norm.inverse((img + 1) / 2))
                           for img, norm in
                           zip(iti_img, [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304]])]]

print('########## load SDO ##########')
#
sdo_dataset = SDODataset("/gss/r.jarolim/data/ch_detection", resolution=4096, n_samples=50)
sdo_iterator = DataLoader(sdo_dataset, batch_size=1, shuffle=False, num_workers=2)

sdo_dates = [parse(b.split('.')[0]) for b in sdo_dataset.basenames]
sdo_lc = []
with torch.no_grad():
    for sdo_img in tqdm(sdo_iterator, total=len(sdo_iterator)):
        sdo_img = sdo_img[0, :-1].detach().cpu().numpy()
        sdo_lc += [[np.mean(norm.inverse((img + 1) / 2))
                    for img, norm in zip(sdo_img, [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304]])]]
        del sdo_img
        gc.collect()

# invert normalization
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))

for ax, lc in zip(axs, np.transpose(sdo_lc)):
    ax.plot(sdo_dates, lc, label='SDO')

for ax, lc in zip(axs, np.transpose(soho_lc)):
    ax.plot(soho_dates, lc, label='SOHO')

for ax, lc in zip(axs, np.transpose(stereo_lc)):
    ax.plot(stereo_dates, lc, label='STEREO')

for ax, lc in zip(axs, np.transpose(soho_iti_lc)):
    ax.plot(soho_dates, lc, label='SOHO - ITI')

for ax, lc in zip(axs, np.transpose(stereo_iti_lc)):
    ax.plot(stereo_dates, lc, label='STEREO - ITI')

[ax.legend() for ax in axs]
axs[0].set_title('171')
axs[1].set_title('193/195')
axs[2].set_title('211/284')
axs[3].set_title('304')

axs[3].set_ylim(None, 150)

fig.tight_layout()
fig.savefig(os.path.join(base_path, 'light_curve.jpg'), dpi=300)
plt.close(fig)
