import logging
import multiprocessing
import os
import threading
from datetime import timedelta, datetime
from multiprocessing.queues import JoinableQueue
from urllib import request

import drms
import numpy as np
import pandas as pd
from astropy.io import fits
from dateutil.parser import parse
from sunpy.map import Map


class HMIContinuumFetcher:

    def __init__(self, ds_path, num_worker_threads=8):
        self.ds_path = ds_path
        os.makedirs(ds_path, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler("{0}/{1}.log".format(ds_path, "info_log")),
                logging.StreamHandler()
            ])

        self.drms_client = drms.Client(email='robert.jarolim@uni-graz.at', verbose=False)
        self.download_queue = JoinableQueue(ctx=multiprocessing.get_context())
        for i in range(num_worker_threads):
            t = threading.Thread(target=self.download_worker)
            t.start()

    def download_worker(self):
        while True:
            header, segment, t = self.download_queue.get()
            logging.info('Download: %s (%d remaining)' % (header['DATE__OBS'], self.download_queue.qsize()))
            try:
                dir = os.path.join(self.ds_path, '%d' % header['WAVELNTH'])
                map_path = os.path.join(dir, '%s.fits' % t.isoformat('T', timespec='seconds'))
                if os.path.exists(map_path):
                    self.download_queue.task_done()
                    continue
                # load map
                url = 'http://jsoc.stanford.edu' + segment
                file_path = os.path.join(self.ds_path, '%s') % segment[1:].replace('/', '-')
                request.urlretrieve(url, filename=file_path)
                hdul = fits.open(file_path)
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
                os.remove(file_path)
                self.download_queue.task_done()
            except Exception as ex:
                logging.info('Download failed: %s (requeue)' % header['DATE__OBS'])
                logging.info(ex)
                self.download_queue.put((header, segment, t))
                self.download_queue.task_done()
                continue

    def fetchDates(self, dates):
        for date in dates:
            try:
                self.fetchData(date)
            except Exception as ex:
                print(ex)
                logging.error('Unable to download: %s' % date.isoformat())
        self.download_queue.join()

    def fetchData(self, time):
        id = time.isoformat()

        logging.info('Start download: %s' % id)
        # query continuum
        time_param = '%sZ' % time.isoformat('_', timespec='seconds')
        ds_hmi = 'hmi.Ic_720s[%s]{continuum}' % time_param
        keys_hmi = self.drms_client.keys(ds_hmi)
        header_hmi, segment_hmi = self.drms_client.query(ds_hmi, key=','.join(keys_hmi), seg='continuum')
        if len(header_hmi) != 1 or np.any(header_hmi.QUALITY != 0):
            raise Exception('No valid data found!')

        for (idx, h), s in zip(header_hmi.iterrows(), segment_hmi.continuum):
            self.download_queue.put((h.to_dict(), s, time))

        logging.info('Finished: %s' % id)

if __name__ == '__main__':
    fetcher = HMIContinuumFetcher(ds_path="/gss/r.jarolim/data/hmi_continuum")
    fetcher.fetchDates([datetime(2010, 3, 29) + i * timedelta(days=1) for i in
                        range((datetime.now() - datetime(2010, 3, 29)) // timedelta(days=1))])
