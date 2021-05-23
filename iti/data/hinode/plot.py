import shutil

import pandas as pd
from sunpy.map import Map
from matplotlib import pyplot as plt
import os
import numpy as np

base_path = '/gss/r.jarolim/data/converted/hinode_continuum_img'
shutil.rmtree(base_path)
os.makedirs(base_path)

df = pd.read_csv('/gss/r.jarolim/data/hinode/file_list.csv', index_col=False, parse_dates=['date'])
#df = df[df.date.dt.month == 12]
df = df[(df.classification == 'feature')]

removed = []
for f in df.file:
    if not os.path.exists(f):
        removed.append(f)

values = []
for f in df.file:
    s_map = Map(f)
    values += [np.std(s_map.data) / s_map.meta['EXPTIME']]
    if np.std(s_map.data / s_map.meta['EXPTIME']) < 2500:
        continue
    s_map.plot()
    plt.savefig(os.path.join(base_path, os.path.basename(f).replace('.fits', '.jpg')))
    plt.close()

plt.hist(values, 100)
plt.savefig('/gss/r.jarolim/data/converted/hinode_continuum_img/hist.jpg')
plt.close()