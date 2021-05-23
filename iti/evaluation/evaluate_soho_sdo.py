import glob
import os
from datetime import datetime

import pylab
from dateutil.parser import parse
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from sunpy.visualization.colormaps import cm
from torch.utils.data import DataLoader

from iti.data.dataset import SOHODataset, SDODataset
from iti.data.editor import PaddingEditor, sdo_norms, soho_norms
from iti.train.model import DiscriminatorMode
from iti.train.trainer import Trainer

import numpy as np

# %% default settings
sdo_shape = 2048
soho_shape = 1024
base_path = "/gss/r.jarolim/iti/soho_sdo_v23"
n_samples = 500

os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)
sdo_cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    plt.get_cmap('gray')
]
# %% load soho data set
soho_path = "/gss/r.jarolim/data/soho/train"
soho_base_names = sorted(
    [os.path.basename(path) for path in glob.glob(os.path.join(soho_path, 'eit_171', "**.fits"), recursive=True)])
soho_base_names = soho_base_names[::len(soho_base_names) // n_samples]
soho_dates = [parse(f[:-4]) for f in soho_base_names]
soho_dataset = SOHODataset(soho_path, basenames=soho_base_names)
soho_dataset.addEditor(PaddingEditor((soho_shape, soho_shape)))
soho_loader = DataLoader(soho_dataset, batch_size=1, num_workers=4, shuffle=False)

# %% load sdo data set
sdo_path = "/gss/r.jarolim/data/ch_detection"
sdo_base_names = sorted(
    [os.path.basename(path) for path in glob.glob(os.path.join(sdo_path, '171', "**.fits"), recursive=True)])
sdo_base_names = sdo_base_names[::len(sdo_base_names) // n_samples]
sdo_dates = [parse(f[:-4]) for f in sdo_base_names]
sdo_dataset = SDODataset(sdo_path, base_names=sdo_base_names)
sdo_dataset.addEditor(PaddingEditor((sdo_shape, sdo_shape)))
sdo_loader = DataLoader(sdo_dataset, batch_size=1, num_workers=2, shuffle=False)

# %% load model
trainer = Trainer(5, 5, upsampling=1, discriminator_mode=DiscriminatorMode.CHANNELS,
                  lambda_diversity=0, norm='in_rs_aff')
trainer.cuda()
iteration = trainer.resume(base_path, 120000)
trainer.eval()

# %% enhance soho images and convert to light curve
sdo_lc = []
iti_lc = []
soho_lc = []
with torch.no_grad():
    for soho_img in tqdm(soho_loader):
        soho_img = soho_img.float().cuda()
        iti_img = trainer.forwardAB(soho_img)
        iti_img[0, -1] = -iti_img[0, -1]
        #
        soho_img = soho_img.detach().cpu().numpy()
        iti_img = iti_img.detach().cpu().numpy()
        #
        iti_channels = []
        for c, norm in enumerate([sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304], sdo_norms['mag']]):
            iti_channels.append([np.mean(norm.inverse((img + 1) / 2)) for img in iti_img[:, c]])
        iti_lc.append(np.stack(iti_channels, 1))
        #
        soho_channels = []
        for c, norm in enumerate(soho_norms.values()):
            soho_channels.append([np.mean(norm.inverse((img + 1) / 2)) for img in soho_img[:, c]])
        soho_lc.append(np.stack(soho_channels, 1))

# %% convert SDO to light curve
for sdo_img in tqdm(sdo_loader):
    sdo_img = sdo_img.detach().cpu().numpy()
    #
    sdo_channels = []
    for c, norm in enumerate([sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304], sdo_norms['mag']]):
        sdo_channels.append([np.mean(norm.inverse((img + 1) / 2)) for img in sdo_img[:, c]])
    sdo_lc.append(np.stack(sdo_channels, 1))

soho_lc = np.concatenate(soho_lc)
sdo_lc = np.concatenate(sdo_lc)
iti_lc = np.concatenate(iti_lc)

#%% remove outlayers
condition = np.abs(soho_lc[:, -1]) < 10
soho_dates = np.array(soho_dates)[condition]
soho_lc = soho_lc[condition]
iti_lc = iti_lc[condition]


#%% load_ss_data
sn_data = np.genfromtxt("/gss/r.jarolim/data/ch_detection/SN_ms_tot_V2.0.csv", delimiter=';')
sn_date = np.array([datetime(year=int(y), month=int(m), day=1) for y, m in zip(sn_data[:, 0], sn_data[:, 1])])

# %% plot
params = {'axes.labelsize': 'large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'legend.fontsize': 'large'}
pylab.rcParams.update(params)

channels = soho_lc.shape[1]
fig, axs = plt.subplots(channels + 1, 1, figsize=[11, 2.5 * channels])
for c, (ax, title) in enumerate(zip(axs, [r'171 $\AA$', r'193 $\AA$', r'211 $\AA$', r'304 $\AA$', 'Magnetogram'])):
    ax.plot(sdo_dates, sdo_lc[:, c], label='SDO', marker='o', markersize=2)
    ax.plot(soho_dates[:-2], soho_lc[:, c][:-2], label='SOHO', marker='o', markersize=2)
    ax.plot(soho_dates[:-2], iti_lc[:, c][:-2], label='ITI', marker='o', markersize=2)
    ax.text(.01, .85, title,
            horizontalalignment='left',
            transform=ax.transAxes, fontsize='x-large')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Intensity')
    ax.set_xlim(min(soho_dates), max(sdo_dates))

axs[0].legend(loc='upper right')
axs[-2].set_ylim(-10, 10)
axs[-2].set_ylabel('Mean Magnetic Flux')

axs[-1].plot(sn_date, sn_data[:, 3], color='black')
axs[-1].set_ylabel('Sunspot Number')
axs[-1].set_xlabel('Date')
axs[-1].text(.01, .85, 'Total Sunspot Number',
        horizontalalignment='left',
        transform=axs[-1].transAxes, fontsize='x-large')
axs[-1].set_xlim(min(soho_dates), max(sdo_dates))

plt.tight_layout()
plt.savefig(os.path.join(base_path, 'evaluation/lightcurve_%06d.jpg' % iteration), dpi=300)
plt.close()
