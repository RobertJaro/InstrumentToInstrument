import os
import shutil
from random import sample

import numpy as np

from iti.data.dataset import get_intersecting_files

soho_path = '/gss/r.jarolim/data/soho_iti2021_prep'
stereo_path = '/gss/r.jarolim/data/stereo_iti2021_prep'

store_path = '/gss/r.jarolim/data/iti_samples'
samples_path = os.path.join(store_path, 'samples')
os.makedirs(samples_path, exist_ok=True)

soho_paths = get_intersecting_files(soho_path, ['171', '195', '284', '304'], months=[12])
stereo_paths = get_intersecting_files(stereo_path, ['171', '195', '284', '304'], months=[12])

soho_sample = sample(np.transpose(soho_paths).tolist(), 1)[0]
stereo_sample = sample(np.transpose(stereo_paths).tolist(), 1)[0]

for f, wl in zip(soho_sample, ['171', '195', '284', '304']):
    dir = os.path.join(samples_path, 'soho', wl)
    os.makedirs(dir, exist_ok=True)
    shutil.copy(f, os.path.join(dir, os.path.basename(f)))

for f, wl in zip(stereo_sample, ['171', '195', '284', '304']):
    dir = os.path.join(samples_path, 'stereo', wl)
    os.makedirs(dir, exist_ok=True)
    shutil.copy(f, os.path.join(dir, os.path.basename(f)))

shutil.make_archive(os.path.join(store_path, 'samples'), 'zip', samples_path)