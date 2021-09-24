import datetime
import gc
import os
from multiprocessing import Pool

import pandas
import torch
from dateutil.parser import parse
from iti.data.editor import soho_norms, sdo_norms, stereo_norms

from iti.data.dataset import SOHODataset, STEREODataset, SDODataset, SOHOHMIDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from iti.prediction.translate import SOHOToSDO, STEREOToSDO, SOHOToSDOEUV

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from iti.prediction.translate import SOHOToSDO, STEREOToSDO

from matplotlib import pyplot as plt

import numpy as np

# init
base_path = '/gss/r.jarolim/iti/euv_comparison'
os.makedirs(base_path, exist_ok=True)
df_path = os.path.join(base_path, 'data.csv')

# init data
df = pandas.DataFrame(columns={'date': [], 'value': [], 'type': [], 'wl': []}) if not os.path.exists(df_path) \
    else pandas.read_csv(df_path, parse_dates=['date'], index_col=0)

# raise Exception('break')
# create translator
df = df[(df.type != 'SOHO') & (df.type != 'SOHO-ITI')]
translator_soho = SOHOToSDOEUV(model_path='/gss/r.jarolim/iti/soho_sdo_euv_v1/generator_AB.pt')
translator_stereo = STEREOToSDO(model_path='/gss/r.jarolim/iti/stereo_v7/generator_AB.pt')

print('########## load SOHO ##########')
soho_dataset = SOHODataset("/gss/r.jarolim/data/soho_iti2021_prep", resolution=1024, n_samples=1000,
                           wavelengths=[171,195,284,304])
soho_iterator = DataLoader(soho_dataset, batch_size=1, shuffle=False, num_workers=12)

soho_dates = [parse(b.split('.')[0]) for b in soho_dataset.basenames]
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

# print('########## load additional SOHO ##########')
# soho_hmi_dataset = SOHOHMIDataset('/gss/r.jarolim/data/soho_iti2021_prep', '/gss/r.jarolim/data/ch_detection/6173',
#                                   resolution=1024, n_samples=1000)
# soho_hmi_iterator = DataLoader(soho_hmi_dataset, batch_size=1, shuffle=False, num_workers=12)
#
# soho_dates = [parse(b.split('.')[0]) for b in soho_hmi_dataset.basenames]
# with torch.no_grad():
#     for soho_img, date in tqdm(zip(soho_hmi_iterator, soho_dates), total=len(soho_hmi_iterator)):
#         soho_img = soho_img.cuda()
#         iti_img = translator_soho.generator(soho_img)
#         # flatten batch + remove magnetogram
#         iti_img = iti_img[0, :-1].detach().cpu().numpy()
#         soho_img = soho_img[0, :-1].detach().cpu().numpy()
#         for img, wl in zip(soho_img, [171, 195, 284, 304]):
#             v = np.mean(soho_norms[wl].inverse((img + 1) / 2))
#             df = df.append({'date': date, 'value': v, 'type': 'SOHO', 'wl': wl}, ignore_index=True)
#         for img, wl in zip(iti_img, [171, 193, 211, 304]):
#             v = np.mean(sdo_norms[wl].inverse((img + 1) / 2))
#             df = df.append({'date': date, 'value': v, 'type': 'SOHO-ITI', 'wl': wl}, ignore_index=True)
#
# df.to_csv(df_path)

print('########## load STEREO ##########')
stereo_dataset = STEREODataset("/gss/r.jarolim/data/stereo_iti2021_prep", n_samples=1000)
stereo_iterator = DataLoader(stereo_dataset, batch_size=1, shuffle=False, num_workers=12)

stereo_dates = [parse(b.split('.')[0]) for b in stereo_dataset.basenames]
with torch.no_grad():
    for stereo_img, date in tqdm(zip(stereo_iterator, stereo_dates), total=len(stereo_iterator)):
        stereo_img = stereo_img.cuda()
        iti_img = translator_stereo.generator(stereo_img)
        iti_img = iti_img[0].detach().cpu().numpy()
        stereo_img = stereo_img[0].detach().cpu().numpy()

        for img, wl in zip(stereo_img, [171, 195, 284, 304]):
            v = np.mean(stereo_norms[wl].inverse((img + 1) / 2))
            df = df.append({'date': date, 'value': v, 'type': 'STEREO', 'wl': wl}, ignore_index=True)
        for img, wl in zip(iti_img, [171, 193, 211, 304]):
            v = np.mean(sdo_norms[wl].inverse((img + 1) / 2))
            df = df.append({'date': date, 'value': v, 'type': 'STEREO-ITI', 'wl': wl}, ignore_index=True)

df.to_csv(df_path)

print('########## load SDO ##########')
#
sdo_dataset = SDODataset("/gss/r.jarolim/data/ch_detection", resolution=4096, n_samples=1000)
sdo_iterator = DataLoader(sdo_dataset, batch_size=1, shuffle=False, num_workers=4, )
sdo_dates = [parse(b.split('.')[0]) for b in sdo_dataset.basenames]

for sdo_img, date in tqdm(zip(sdo_iterator, sdo_dates), total=len(sdo_iterator)):
    sdo_img = sdo_img[0, :-1].detach().cpu().numpy()
    for img, wl in zip(sdo_img, [171, 193, 211, 304]):
        v = np.mean(sdo_norms[wl].inverse((img + 1) / 2))
        df = df.append({'date': date, 'value': v, 'type': 'SDO', 'wl': wl}, ignore_index=True)

df.to_csv(df_path)

df = df.sort_values('date')
# invert normalization
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
lines = []

for ax, d in zip(axs, [df[(df.type == 'SDO') & (df.wl == wl)] for wl in [171, 193, 211, 304]]):
    line = ax.plot(d.date, d.value, label='SDO', zorder=10)

lines += line

for ax, d in zip(axs, [df[(df.type == 'SOHO') & (df.wl == wl)] for wl in [171, 195, 284, 304]]):
    line = ax.plot(d.date, d.value, label='SOHO', zorder=6)

lines += line

for ax, d in zip(axs, [df[(df.type == 'STEREO') & (df.wl == wl)] for wl in [171, 195, 284, 304]]):
    line = ax.plot(d.date, d.value, label='STEREO', zorder=7)

lines += line

for ax, d in zip(axs, [df[(df.type == 'SOHO-ITI') & (df.wl == wl)] for wl in [171, 193, 211, 304]]):
    line = ax.plot(d.date, d.value, label='SOHO - ITI', zorder=8)

lines += line

for ax, d in zip(axs, [df[(df.type == 'STEREO-ITI') & (df.wl == wl)] for wl in [171, 193, 211, 304]]):
    line = ax.plot(d.date, d.value, label='STEREO - ITI', zorder=9)

lines += line

axs[0].set_title('171')
axs[1].set_title('193/195')
axs[2].set_title('211/284')
axs[3].set_title('304')

axs[3].set_ylim(None, 150)

fig.tight_layout()
fig.savefig(os.path.join(base_path, 'light_curve.jpg'), dpi=300)
plt.close(fig)

fig = plt.figure(figsize=(8, .8))
fig.legend(lines, ['SDO', 'SOHO', 'STEREO', 'SOHO - ITI', 'STEREO - ITI'], loc='lower center', ncol=3, fontsize=14)
fig.savefig(os.path.join(base_path, 'lc_legend.jpg'), dpi=300)
plt.close(fig)


#################################################
# CORRELATION
#################################################
def to_float(dates,ref):
    return np.array([(d - ref).total_seconds() for d in dates])

def correlation(dates_1, values_1, dates_2, values_2):
    min_date = max(min(dates_1), min(dates_2))
    max_date = min(max(dates_1), max(dates_2))
    interp_dates = [datetime.datetime(year=min_date.year, month=min_date.month, day=min_date.day) + i * datetime.timedelta(days=1)
                    for i in range((max_date - min_date) // datetime.timedelta(days=1))]
    #
    interp_dates = np.array(interp_dates)
    dates_1, values_1 = np.array(dates_1), np.array(values_1)
    dates_2, values_2 = np.array(dates_2), np.array(values_2)
    #
    condition = np.logical_and(dates_1 > min_date, dates_1 < max_date)
    values_1_interp = np.interp(to_float(interp_dates, min_date), to_float(dates_1[condition], min_date), values_1[condition])
    condition = np.logical_and(dates_2 > min_date, dates_2 < max_date)
    values_2_interp = np.interp(to_float(interp_dates, min_date), to_float(dates_2[condition], min_date), values_2[condition])
    #
    w = 60
    smooth_1 = np.convolve(values_1_interp, np.ones((w,)) / w, mode='valid')
    smooth_2 = np.convolve(values_2_interp, np.ones((w,)) / w, mode='valid')
    interp_dates = interp_dates[w//2:-(w//2 - 1)]
    # smooth_1 = (smooth_1 - np.mean(smooth_1)) / np.std(smooth_1)
    # smooth_2 = (smooth_2 - np.mean(smooth_2)) / np.std(smooth_2)
    #
    mse = np.mean(np.abs(smooth_1 - smooth_2)) / np.mean(smooth_1)
    corr = np.corrcoef(smooth_1, smooth_2)
    lin_fit = np.polyfit(smooth_1, smooth_2, 1)
    #
    return corr[0, 1], mse, lin_fit, (interp_dates, smooth_1, smooth_2)

correlation_coeff = {'SDO':{},'SOHO':{}, 'STEREO':{},'SOHO-ITI':{},'STEREO-ITI':{}}
mse_loss = {'SDO':{},'SOHO':{}, 'STEREO':{},'SOHO-ITI':{},'STEREO-ITI':{}}
fit_coeff = {'SDO':{},'SOHO':{}, 'STEREO':{},'SOHO-ITI':{},'STEREO-ITI':{}}
fig, axs = plt.subplots(4, 4, figsize=(8, 8))

for row, wl in zip(axs, [171, 193, 211, 304]):
    sdo = df[(df.type == 'SDO') & (df.wl == wl)]
    corr, mse, coeff, (dates, sdo_lc, _) = correlation([d.to_pydatetime() for d in sdo.date], sdo.value,
                                            [d.to_pydatetime() for d in sdo.date], sdo.value)
    # ax.plot([min(sdo_lc), max(sdo_lc)], [min(sdo_lc), max(sdo_lc)], color='black')
    correlation_coeff['SDO'][wl] = round(corr, 3)
    mse_loss['SDO'][wl] = round(mse, 3)
    fit_coeff['SDO'][wl] = np.round(coeff, 3)

for ax, wl, wl_soho in zip(axs[:, 0], [171, 193, 211, 304], [171, 195, 284, 304]):
    sdo = df[(df.type == 'SDO') & (df.wl == wl)]
    ref = df[(df.type == 'SOHO') & (df.wl == wl_soho)]
    corr, mse, coeff, (dates, sdo_lc, ref_lc) = correlation([d.to_pydatetime() for d in sdo.date], sdo.value,
                                                [d.to_pydatetime() for d in ref.date], ref.value)
    ax.scatter(sdo_lc, ref_lc)
    ax.plot(sdo_lc, np.poly1d(coeff)(sdo_lc), '--k', label='SOHO')
    correlation_coeff['SOHO'][wl] = round(corr, 3)
    mse_loss['SOHO'][wl] = round(mse, 3)
    fit_coeff['SOHO'][wl] = np.round(coeff, 3)

for ax, wl, wl_stereo in zip(axs[:, 1], [171, 193, 211, 304], [171, 195, 284, 304]):
    sdo = df[(df.type == 'SDO') & (df.wl == wl)]
    ref = df[(df.type == 'STEREO') & (df.wl == wl_stereo)]
    corr, mse, coeff, (dates, sdo_lc, ref_lc) = correlation([d.to_pydatetime() for d in sdo.date], sdo.value,
                                                [d.to_pydatetime() for d in ref.date], ref.value)
    ax.scatter(sdo_lc, ref_lc)
    ax.plot(sdo_lc, np.poly1d(coeff)(sdo_lc), '--k', label='STEREO')
    correlation_coeff['STEREO'][wl] = round(corr, 3)
    mse_loss['STEREO'][wl] = round(mse, 3)
    fit_coeff['STEREO'][wl] = np.round(coeff, 3)

for ax, wl in zip(axs[:, 2], [171, 193, 211, 304]):
    sdo = df[(df.type == 'SDO') & (df.wl == wl)]
    ref = df[(df.type == 'SOHO-ITI') & (df.wl == wl)]
    corr, mse, coeff, (dates, sdo_lc, ref_lc) = correlation([d.to_pydatetime() for d in sdo.date], sdo.value,
                                                [d.to_pydatetime() for d in ref.date], ref.value)
    ax.scatter(sdo_lc, ref_lc)
    ax.plot(sdo_lc, np.poly1d(coeff)(sdo_lc), '--k', label='SOHO-ITI')
    correlation_coeff['SOHO-ITI'][wl] = round(corr, 3)
    mse_loss['SOHO-ITI'][wl] = round(mse, 3)
    fit_coeff['SOHO-ITI'][wl] = np.round(coeff, 3)

for ax, wl in zip(axs[:, 3], [171, 193, 211, 304]):
    sdo = df[(df.type == 'SDO') & (df.wl == wl)]
    ref = df[(df.type == 'STEREO-ITI') & (df.wl == wl)]
    corr, mse, coeff, (dates, sdo_lc, ref_lc) = correlation([d.to_pydatetime() for d in sdo.date], list(sdo.value),
                                                [d.to_pydatetime() for d in ref.date], list(ref.value))
    ax.scatter(sdo_lc, ref_lc)
    ax.plot(sdo_lc, np.poly1d(coeff)(sdo_lc), '--k', label='STEREO-ITI')
    # ax.plot(dates, ref_lc, label='STEREO - ITI')
    correlation_coeff['STEREO-ITI'][wl] = round(corr, 3)
    mse_loss['STEREO-ITI'][wl] = round(mse, 3)
    fit_coeff['STEREO-ITI'][wl] = np.round(coeff, 3)


[ax.legend() for ax in axs]
axs[0, 0].set_title('171')
axs[1, 0].set_title('193/195')
axs[2, 0].set_title('211/284')
axs[3, 0].set_title('304')

#axs[3].set_ylim(None, 150)

fig.tight_layout()
fig.savefig(os.path.join(base_path, 'correlation.jpg'), dpi=300)
plt.close(fig)

pandas.DataFrame(correlation_coeff)
pandas.DataFrame(mse_loss)
pandas.DataFrame(fit_coeff)