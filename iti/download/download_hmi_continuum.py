import argparse
import logging
import multiprocessing
import os
from datetime import datetime, timedelta
from urllib import request

import drms
import numpy as np
import pandas as pd
from astropy.io import fits
from sunpy.io.fits import header_to_fits
from sunpy.util import MetaDict


class HMIContinuumDownloader:

    def __init__(self, ds_path, email, num_worker_threads=4, ignore_quality=False, series='hmi.Ic_720s'):
        self.series = series
        self.ignore_quality = ignore_quality
        self.ds_path = ds_path
        self.n_workers = num_worker_threads
        os.makedirs(ds_path, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler("{0}/{1}.log".format(ds_path, "info_log")),
                logging.StreamHandler()
            ])

        self.drms_client = drms.Client(email=email, verbose=False)

    def download(self, data):
        header, segment, t = data
        map_path = os.path.join(self.ds_path, '%s.fits' % t.isoformat('T', timespec='seconds'))
        if os.path.exists(map_path):
            return map_path
        # load map
        url = 'http://jsoc.stanford.edu' + segment
        request.urlretrieve(url, filename=map_path)

        header['DATE_OBS'] = header['DATE__OBS']
        header = header_to_fits(MetaDict(header))
        with fits.open(map_path, 'update') as f:
            hdr = f[1].header
            for k, v in header.items():
                if pd.isna(v):
                    continue
                hdr[k] = v
            f.verify('silentfix')

        return map_path

    def fetchDates(self, dates):
        header_info = []
        logging.info('Fetch header information')
        for date in dates:
            try:
                header_info += self.fetchData(date)
            except Exception as ex:
                print(ex)
                logging.error('Unable to download: %s' % date.isoformat())

        logging.info('Download data')
        with multiprocessing.Pool(self.n_workers) as p:
            files = p.map(self.download, header_info)
        return files

    def fetchData(self, time):
        # query
        time_param = '%sZ' % time.isoformat('_', timespec='seconds')
        ds_hmi = '%s[%s]{continuum}' % (self.series, time_param)
        keys_hmi = self.drms_client.keys(ds_hmi)
        header_hmi, segment_hmi = self.drms_client.query(ds_hmi, key=','.join(keys_hmi), seg='continuum')
        if len(header_hmi) != 1 or (np.any(header_hmi.QUALITY != 0) and not self.ignore_quality):
            raise Exception('No valid data found!')

        data = [(h.to_dict(), s, time) for (idx, h), s in zip(header_hmi.iterrows(), segment_hmi.continuum)]
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download SDO/HMI data')
    parser.add_argument('--download_dir', type=str, help='path to the download directory.')
    parser.add_argument('--email', type=str, help='registered email address for JSOC.')
    args = parser.parse_args()

    fetcher = HMIContinuumDownloader(ds_path=args.download_dir, email=args.email)
    for y in range(2010, 2023):
        dates = [datetime(y, 1, 1) + i * timedelta(days=1) for i in
                 range((datetime(y + 1, 1, 1) - datetime(y, 1, 1)) // timedelta(days=1))]
        dates = [d for d in dates if d.month in [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
        fetcher.fetchDates(dates)
