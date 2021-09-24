import os
from random import sample

import pandas as pd

from iti.download.download_hmi_continuum import HMIContinuumDownloader

base_path = '/gss/r.jarolim/data/hmi_hinode_comparison_qs'
os.makedirs(base_path, exist_ok=True)

df = pd.read_csv('/gss/r.jarolim/data/hinode/file_list.csv', index_col=False, parse_dates=['date'])
test_df = df[df.date.dt.month.isin([11, 12])]
test_df = test_df[test_df.classification == 'quiet']

hinode_dates = [d.to_pydatetime() for d in test_df.date]
hinode_dates = sample(hinode_dates, 10)

fetcher = HMIContinuumDownloader(ds_path=base_path, num_worker_threads=4)
fetcher.fetchDates(hinode_dates)