from datetime import datetime, timedelta
from glob import glob

import os
from random import random, sample

from sunpy.map import Map
from tqdm import tqdm

files = sorted(glob('/localdata/USER/rja/stereo_iti2021/**/*.fits'))


start_date = datetime(2006, 11, 20)
for i in tqdm(range((datetime.now() - start_date) // timedelta(days=1))):
    check_date = start_date + timedelta(days=i)
    files_day = [f for f in files if os.path.basename(f)[:10] == '%04d-%02d-%02d' % (check_date.year, check_date.month, check_date.day)]
    if len(files_day) == 0:
        continue
    basenames = set([os.path.basename(f) for f in files_day])
    selected_basename = sample(basenames, 1)[0]
    [os.remove(f) for f in files_day if os.path.basename(f) != selected_basename]


invalid = []
for f in tqdm(files):
    try:
        Map(f)
    except:
        print('invalid')
        invalid.append(f)