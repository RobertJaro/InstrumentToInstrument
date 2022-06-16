import datetime
import os

import pandas
import torch
from dateutil.parser import parse
from iti.data.editor import soho_norms, sdo_norms, stereo_norms, get_auto_calibration_table

from iti.data.dataset import SOHODataset, STEREODataset, SDODataset, get_intersecting_files, AIADataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from iti.translate import SOHOToSDOEUV



from iti.translate import STEREOToSDO

from matplotlib import pyplot as plt

import numpy as np

# init
base_path = '/gpfs/gpfs0/robert.jarolim/iti/euv_calibration'
os.makedirs(base_path, exist_ok=True)
df_path = os.path.join(base_path, 'data.csv')

# init data
df = pandas.DataFrame(columns={'date': [], 'value': [], 'type': [], 'wl': []})

print('########## load SDO ##########')
#
sdo_files = get_intersecting_files("/gpfs/gpfs0/robert.jarolim/data/iti/sdo", ['171', '193', '211', '304', '6173'], ext='.fits')
sdo_files = np.array(sdo_files)[:, ::100].tolist()
#
sdo_dataset = AIADataset(sdo_files[3], 304, resolution=4096, calibration=None)
sdo_iterator = DataLoader(sdo_dataset, batch_size=1, shuffle=False, num_workers=4, )
sdo_dates = [parse(sdo_dataset.getId(i)) for i in range(len(sdo_dataset))]

for sdo_img, date in tqdm(zip(sdo_iterator, sdo_dates), total=len(sdo_iterator)):
    sdo_img = sdo_img[0].detach().cpu().numpy()
    for img, wl in zip(sdo_img, [304]):
        v = np.mean(sdo_norms[wl].inverse((img + 1) / 2))
        df = df.append({'date': date, 'value': v, 'type': 'SDO', 'wl': wl}, ignore_index=True)


print('########## load SDO aiapy ##########')
#
sdo_dataset = AIADataset(sdo_files[3], 304, resolution=4096, calibration='aiapy')
sdo_iterator = DataLoader(sdo_dataset, batch_size=1, shuffle=False, num_workers=4, )
sdo_dates = [parse(sdo_dataset.getId(i)) for i in range(len(sdo_dataset))]

for sdo_img, date in tqdm(zip(sdo_iterator, sdo_dates), total=len(sdo_iterator)):
    sdo_img = sdo_img[0].detach().cpu().numpy()
    for img, wl in zip(sdo_img, [304]):
        v = np.mean(sdo_norms[wl].inverse((img + 1) / 2))
        df = df.append({'date': date, 'value': v, 'type': 'SDO-aiapy', 'wl': wl}, ignore_index=True)


print('########## load SDO autocalibration ##########')
#
sdo_dataset = AIADataset(sdo_files[3], 304, resolution=4096, calibration='auto')
sdo_iterator = DataLoader(sdo_dataset, batch_size=1, shuffle=False, num_workers=4, )
sdo_dates = [parse(sdo_dataset.getId(i)) for i in range(len(sdo_dataset))]

for sdo_img, date in tqdm(zip(sdo_iterator, sdo_dates), total=len(sdo_iterator)):
    sdo_img = sdo_img[0].detach().cpu().numpy()
    for img, wl in zip(sdo_img, [304]):
        v = np.mean(sdo_norms[wl].inverse((img + 1) / 2))
        df = df.append({'date': date, 'value': v, 'type': 'SDO-auto', 'wl': wl}, ignore_index=True)

df.to_csv(df_path)

df = df.sort_values('date')
# invert normalization
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 2))

d = df[(df.type == 'SDO')]
ax.plot(d.date, d.value, label='SDO original')

d = df[(df.type == 'SDO-aiapy')]
ax.plot(d.date, d.value, label='SDO calibrated')

d = df[(df.type == 'SDO-auto')]
ax.plot(d.date, d.value, label='SDO auto-calibration')

ax.set_title('304')

# axs[3].set_ylim(None, 150)
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(base_path, '304_light_curve.jpg'), dpi=300)
plt.close(fig)

table = get_auto_calibration_table()
plt.plot(table['DATE'], table['0304'])
plt.savefig('/gpfs/gpfs0/robert.jarolim/iti/euv_calibration/calibration_median.jpg')
plt.close()