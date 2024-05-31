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
    """
    Class to download HMI continuum data from JSOC.

    Args:
        ds_path (str): Path to the directory where the downloaded data should be stored.
        email (str): Email address for JSOC registration.
        num_worker_threads (int): Number of worker threads for parallel download.
        ignore_quality (bool): If True, data with quality flag != 0 will be downloaded.
        series (str): Series name of the HMI continuum data.
    """
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
        """
        Download the data from JSOC.

        Args:
            data (tuple): Tuple containing the header, segment and time information.

        Returns:
            str: Path to the downloaded file.
        """
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
        """
        Fetch the data for the given dates.

        Args:
            dates (list): List of dates for which the data should be downloaded.

        Returns:
            list: List of downloaded files.
        """
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
        """
        Fetch the data for the given time.

        Args:
            time (datetime): Time for which the data should be downloaded.

        Returns:
            list: List of tuples containing the header, segment and time information.
        """
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
