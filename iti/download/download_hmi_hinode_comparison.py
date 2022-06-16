import os

import pandas as pd

from iti.download.download_hmi_continuum import HMIContinuumDownloader

base_path = '/gpfs/gpfs0/robert.jarolim/data/iti/hmi_hinode_comparison_dcon'
os.makedirs(base_path, exist_ok=True)

df = pd.read_csv('/gpfs/gpfs0/robert.jarolim/data/iti/hinode_file_list.csv', index_col=False, parse_dates=['date'])
test_df = df[df.date.dt.month.isin([11, 12])]
test_df = test_df[test_df.classification == 'feature']

hinode_dates = [d.to_pydatetime() for d in test_df.date]
# hinode_dates = sample(hinode_dates, 10)

fetcher = HMIContinuumDownloader(ds_path=base_path, num_worker_threads=4, series='hmi.Ic_45s_dcon', email='robert.jarolim@uni-graz.at')
fetcher.fetchDates(hinode_dates)
