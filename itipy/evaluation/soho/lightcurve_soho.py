import datetime
import os

import pandas
import pandas as pd
import torch
from dateutil.parser import parse
from itipy.train.util import skip_invalid
from sunpy.visualization.colormaps import cm

from itipy.data.editor import soho_norms, sdo_norms, stereo_norms

from itipy.data.dataset import SOHODataset, STEREODataset, SDODataset, get_intersecting_files
from torch.utils.data import DataLoader
from tqdm import tqdm

from itipy.translate import SOHOToSDOEUV, SOHOToSDO

from itipy.translate import STEREOToSDO

from matplotlib import pyplot as plt

import numpy as np

# init
base_path = '/gpfs/gpfs0/robert.jarolim/iti/euv_comparison_v2'
os.makedirs(base_path, exist_ok=True)
df_path = os.path.join(base_path, 'data.csv')

# init data
df = pandas.DataFrame(columns={'date': [], 'value': [], 'type': [], 'wl': []}) if not os.path.exists(df_path) \
    else pandas.read_csv(df_path, parse_dates=['date'], index_col=0)

# raise Exception('break')
# create translator
# df = df[(df.type != 'STEREO') & (df.type != 'STEREO-ITI')]
df = df[(df.type != 'SOHO') & (df.type != 'SOHO-ITI')]
# df = df[(df.type != 'SDO')]
translator_soho = SOHOToSDO(model_path='/gpfs/gpfs0/robert.jarolim/iti/soho_sdo_v4/generator_AB.pt')
translator_stereo = STEREOToSDO(model_path='/gpfs/gpfs0/robert.jarolim/iti/stereo_to_sdo_v1/generator_AB.pt')

def filter_files(files):
    dates = [parse(os.path.basename(f).split('.')[0])for f in files[0]]
    df = pd.DataFrame({'date':dates, 'idx': list(range(len(files[0])))})
    df = df.set_index('date').groupby(pd.Grouper(freq='5D')).first()
    idx = df[~pd.isna(df['idx'])]['idx'].astype(np.int)
    return np.array(files)[:, idx].tolist()

print('########## load SOHO ##########')
soho_files = get_intersecting_files("/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep", [171, 195, 284, 304, 'mag', ],
                               ext='.fits')
soho_files = filter_files(soho_files)
soho_dataset = SOHODataset(soho_files, resolution=1024, wavelengths=None)
soho_iterator = DataLoader(soho_dataset, batch_size=1, shuffle=False, num_workers=12)

soho_dates = [parse(soho_dataset.getId(i)) for i in range(len(soho_dataset))]
with torch.no_grad():
    for soho_img, date in tqdm(zip(soho_iterator, soho_dates), total=len(soho_iterator)):
        soho_img = soho_img.cuda()
        iti_img = translator_soho.generator(soho_img)
        # flatten batch + remove magnetogram
        iti_img = iti_img[0].detach().cpu().numpy()
        soho_img = soho_img[0].detach().cpu().numpy()
        #
        for img, wl in zip(soho_img, [171, 195, 284, 304]):
            v = np.mean(soho_norms[wl].inverse((img + 1) / 2))
            df = df.append({'date': date, 'value': v, 'type': 'SOHO', 'wl': wl}, ignore_index=True)
        for img, wl in zip(iti_img, [171, 193, 211, 304]):
            v = np.mean(sdo_norms[wl].inverse((img + 1) / 2))
            df = df.append({'date': date, 'value': v, 'type': 'SOHO-ITI', 'wl': wl}, ignore_index=True)

df.to_csv(df_path)

# print('########## load STEREO ##########')
# stereo_files = get_intersecting_files("/gpfs/gpfs0/robert.jarolim/data/iti/stereo_iti2021_prep",
#                                     [171, 195, 284, 304, ], ext='.fits')
# stereo_files = filter_files(stereo_files)
# stereo_dataset = STEREODataset(stereo_files)
# stereo_iterator = DataLoader(stereo_dataset, batch_size=1, shuffle=False, num_workers=12)
#
# stereo_dates = [parse(stereo_dataset.getId(i)) for i in range(len(stereo_dataset))]
# with torch.no_grad():
#     for stereo_img, date in tqdm(zip(stereo_iterator, stereo_dates), total=len(stereo_iterator)):
#         stereo_img = stereo_img.cuda()
#         iti_img = translator_stereo.generator(stereo_img)
#         iti_img = iti_img[0].detach().cpu().numpy()
#         stereo_img = stereo_img[0].detach().cpu().numpy()
#         #
#         for img, wl in zip(stereo_img, [171, 195, 284, 304]):
#             v = np.mean(stereo_norms[wl].inverse((img + 1) / 2))
#             df = df.append({'date': date, 'value': v, 'type': 'STEREO', 'wl': wl}, ignore_index=True)
#         for img, wl in zip(iti_img, [171, 193, 211, 304]):
#             v = np.mean(sdo_norms[wl].inverse((img + 1) / 2))
#             df = df.append({'date': date, 'value': v, 'type': 'STEREO-ITI', 'wl': wl}, ignore_index=True)
#         #
#         fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#         axs[0].imshow(stereo_img[-1], cmap=cm.sdoaia304, vmin=-1, vmax=1)
#         axs[1].imshow(iti_img[-1], cmap=cm.sdoaia304, vmin=-1, vmax=1)
#         axs[0].set_axis_off(), axs[1].set_axis_off()
#         plt.tight_layout(pad=0)
#         plt.savefig(os.path.join(base_path, 'stereo_%s.jpg' % date.isoformat('T')))
#         plt.close()
#
# df.to_csv(df_path)

# print('########## load SDO ##########')
# sdo_files = get_intersecting_files("/gpfs/gpfs0/robert.jarolim/data/iti/sdo", ['171', '193', '211', '304', '6173'], ext='.fits')
# sdo_files = filter_files(sdo_files)
#
# sdo_dataset = SDODataset(sdo_files, resolution=4096)
# sdo_iterator = DataLoader(sdo_dataset, batch_size=1, shuffle=False, num_workers=12, )
#
# sdo_dates = [parse(sdo_dataset.getId(i)) for i in range(len(sdo_dataset))]
# for sdo_img, date in tqdm(skip_invalid(zip(sdo_iterator, sdo_dates)), total=len(sdo_iterator)):
#     sdo_img = sdo_img[0, :-1].detach().cpu().numpy()
#     for img, wl in zip(sdo_img, [171, 193, 211, 304]):
#         v = np.mean(sdo_norms[wl].inverse((img + 1) / 2))
#         df = df.append({'date': date, 'value': v, 'type': 'SDO', 'wl': wl}, ignore_index=True)
#
# df.to_csv(df_path)

df = df.sort_values('date')
df = df.set_index('date')

eit_calibration = {'171': [113.69278, 40.340622], '195': [60.60053, 31.752682], '284': [4.7249465, 3.9555929], '304': [64.73511, 26.619505]}
secchi_calibration = {'171': [167.85056, 63.00634], '195': [63.2936, 36.515015], '284': [15.409555, 34.620598], '304': [450.23215, 152.41183]}
aia_calibration = {'171': [148.90274, 62.101795], '193': [146.01889, 71.47675], '211': [44.460854, 27.592617], '304': [46.21493, 18.522688]}

channel_mapping = {s:t for s,t in zip([171, 195, 284, 304], [171, 193, 211, 304])}

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
lines = []
n_rolling = '60D'

for ax, d in zip(axs, [df[(df.type == 'SDO') & (df.wl == wl)] for wl in [171, 193, 211, 304]]):
    values = d.value.rolling(n_rolling, center=True).median()
    std = d.value.rolling(n_rolling, center=True).std()
    line = ax.plot(d.index, values, label='SDO', zorder=10)
    ax.fill_between(d.index, values - std, values + std, zorder=10, color='blue', alpha=0.2)

lines += line

for ax, wl in zip(axs, [171, 195, 284, 304]):
    d = df[(df.type == 'SOHO') & (df.wl == wl)]
    values = d.rolling(n_rolling, center=True).median().value
    eit_mean, eit_std = eit_calibration[str(wl)]
    aia_mean, aia_std = aia_calibration[str(channel_mapping[wl])]
    values = (np.array(values) - eit_mean) * (aia_std / eit_std) + aia_mean
    line = ax.plot(d.index, values, label='SOHO', zorder=6)

lines += line

for ax, wl in zip(axs, [171, 195, 284, 304]):
    d = df[(df.type == 'STEREO') & (df.wl == wl)]
    values = d.value.rolling(n_rolling, center=True).median()
    secchi_mean, secchi_std = secchi_calibration[str(wl)]
    aia_mean, aia_std = aia_calibration[str(channel_mapping[wl])]
    values = (np.array(values) - secchi_mean) * (aia_std / secchi_std) + aia_mean
    line = ax.plot(d.index, values, label='STEREO', zorder=7)

lines += line

for ax, d in zip(axs, [df[(df.type == 'SOHO-ITI') & (df.wl == wl)] for wl in [171, 193, 211, 304]]):
    values = d.value.rolling(n_rolling, center=True).median()
    line = ax.plot(d.index, values, label='SOHO - ITI', zorder=8)

lines += line

for ax, d in zip(axs, [df[(df.type == 'STEREO-ITI') & (df.wl == wl)] for wl in [171, 193, 211, 304]]):
    values = d.value.rolling(n_rolling, center=True).median()
    line = ax.plot(d.index, values, label='STEREO - ITI', zorder=9)

lines += line

axs[0].set_title('171')
axs[1].set_title('193/195')
axs[2].set_title('211/284')
axs[3].set_title('304')

fig.tight_layout()
fig.savefig(os.path.join(base_path, 'light_curve.jpg'), dpi=300)
plt.close(fig)

fig = plt.figure(figsize=(8, .8))
fig.legend(lines, ['SDO', 'SOHO - baseline', 'STEREO - baseline', 'SOHO - ITI', 'STEREO - ITI'], loc='lower center', ncol=3, fontsize=14)
fig.savefig(os.path.join(base_path, 'lc_legend.jpg'), dpi=300)
plt.close(fig)

df['date'] = df.index
maes = []
ccs = []
with open(os.path.join(base_path, 'stereo_evaluation.txt'), 'w') as f:
    for wl in [171, 195, 284, 304]:
        secchi_mean, secchi_std = secchi_calibration[str(wl)]
        aia_wl = channel_mapping[wl]
        aia_mean, aia_std = aia_calibration[str(aia_wl)]
        #
        d = df[(df.type == 'STEREO') & (df.wl == wl)]
        stereo_values = d.groupby(pd.Grouper(key='date',freq='M')).median()
        #
        d = df[(df.type == 'STEREO-ITI') & (df.wl == aia_wl)]
        iti_values = d.groupby(pd.Grouper(key='date',freq='M')).median()
        #
        d = df[(df.type == 'SDO') & (df.wl == aia_wl)]
        sdo_values = d.groupby(pd.Grouper(key='date',freq='M')).median()
        #
        sdo_med = []
        stereo_med = []
        calibrated_med = []
        iti_med = []
        for d, stereo_v, iti_v in zip(stereo_values.index, stereo_values.value, iti_values.value):
            sdo_v = sdo_values[sdo_values.index == d].value
            if len(sdo_v) == 0:
                continue
            calibrated_v = (np.array(stereo_v) - secchi_mean) * (aia_std / secchi_std) + aia_mean
            sdo_med += [float(sdo_v)]
            stereo_med += [stereo_v]
            iti_med += [iti_v]
            calibrated_med += [calibrated_v]
        #
        print(wl, 'MAE', file=f)
        mae_calibrated = np.nanmean(np.abs(np.array(sdo_med) - np.array(calibrated_med)))
        print('calibrated', mae_calibrated, file=f)
        print('original', np.nanmean(np.abs(np.array(sdo_med) - np.array(stereo_med))), file=f)
        mae_iti = np.nanmean(np.abs(np.array(sdo_med) - np.array(iti_med)))
        print('iti', mae_iti, file=f)
        #
        print(wl, 'CC', file=f)
        cond = ~np.isnan(stereo_med) & ~np.isnan(sdo_med)
        cc_calibrated = np.corrcoef(np.array(sdo_med)[cond], np.array(calibrated_med)[cond])[0, 1]
        print('calibrated', cc_calibrated, file=f)
        print('original', np.corrcoef(np.array(sdo_med)[cond], np.array(stereo_med)[cond])[0, 1], file=f)
        cc_iti = np.corrcoef(np.array(sdo_med)[cond], np.array(iti_med)[cond])[0, 1]
        print('iti', cc_iti, file=f)
        maes += [(mae_calibrated, mae_iti)]
        ccs += [(cc_calibrated, cc_iti)]
    print('AVG', 'MAE', file=f)
    print('calibrated', np.mean(np.array(maes)[:, 0]), file=f)
    print('iti', np.mean(np.array(maes)[:, 1]), file=f)
    print('AVG', 'CC', file=f)
    print('calibrated', np.mean(np.array(ccs)[:, 0]), file=f)
    print('iti', np.mean(np.array(ccs)[:, 1]), file=f)