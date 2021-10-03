import logging
import multiprocessing
import os
from datetime import datetime
from urllib.request import urlopen

import drms
import numpy as np
import pandas as pd
from astropy.io.fits import HDUList
from sunpy.map import Map


class HMIContinuumDownloader:

    def __init__(self, ds_path, num_worker_threads=8, ignore_quality=False, series='hmi.Ic_720s'):
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

        self.drms_client = drms.Client(email='robert.jarolim@uni-graz.at', verbose=False)

    def download(self, data):
        header, segment, t = data
        dir = os.path.join(self.ds_path, '%d' % header['WAVELNTH'])
        map_path = os.path.join(dir, '%s.fits' % t.isoformat('T', timespec='seconds'))
        if os.path.exists(map_path):
            return map_path
        # load map
        url = 'http://jsoc.stanford.edu' + segment
        with urlopen(url) as url_request:
            fits_data = url_request.read()
        hdul = HDUList.fromstring(fits_data)
        hdul.verify('silentfix')
        data = hdul[1].data
        header = {k: v for k, v in header.items() if not pd.isna(v)}
        header['DATE_OBS'] = header['DATE__OBS']
        s_map = Map(data, header)
        os.makedirs(dir, exist_ok=True)
        map_path = os.path.join(dir, '%s.fits' % t.isoformat('T', timespec='seconds'))
        if os.path.exists(map_path):
            os.remove(map_path)
        s_map.save(map_path)
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
    fetcher = HMIContinuumDownloader(ds_path="/gss/r.jarolim/data/hmi_continuum")
    # fetcher.fetchDates([datetime(2010, 3, 29) + i * timedelta(days=1) for i in
    #                     range((datetime.now() - datetime(2010, 3, 29)) // timedelta(days=1))])
    fetcher.fetchDates([datetime(2014, 12, 22)])
