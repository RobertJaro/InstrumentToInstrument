import argparse
import logging
import multiprocessing
import os
from datetime import timedelta, datetime
from urllib import request


import drms
import numpy as np
import pandas as pd
from astropy.io import fits
from sunpy.io._fits import header_to_fits
from sunpy.util import MetaDict


class SDODownloader:
    """
    Class to download SDO data from JSOC.

    Args:
        base_path (str): Path to the directory where the downloaded data should be stored.
        email (str): Email address for JSOC registration.
        wavelengths (list): List of wavelengths to download.
        n_workers (int): Number of worker threads for parallel download.
    """
    def __init__(self, base_path, email, wavelengths=['131', '171', '193', '211', '304', '335'], n_workers=5):
        self.ds_path = base_path
        self.wavelengths = [str(wl) for wl in wavelengths]
        self.n_workers = n_workers
        #[os.makedirs(os.path.join(base_path, wl), exist_ok=True) for wl in self.wavelengths + ['6173']]
        [os.makedirs(os.path.join(base_path, wl), exist_ok=True) for wl in self.wavelengths]
        self.drms_client = drms.Client(email=email, verbose=False)

    def download(self, sample):
        """
        Download the data from JSOC.

        Args:
            sample (tuple): Tuple containing the header, segment and time information.

        Returns:
            str: Path to the downloaded file.
        """
        header, segment, t = sample
        try:
            dir = os.path.join(self.ds_path, '%d' % header['WAVELNTH'])
            map_path = os.path.join(dir, '%s.fits' % t.isoformat('T', timespec='seconds'))
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
        except Exception as ex:
            logging.info('Download failed: %s (requeue)' % header['DATE__OBS'])
            logging.info(ex)
            raise ex

    def downloadDate(self, date):
        """
        Download the data for the given date.

        Args:
            date (datetime): The date for which the data should be downloaded.

        Returns:
            list: List of paths to the downloaded files.
        """
        id = date.isoformat()

        logging.info('Start download: %s' % id)
        # query Magnetogram
        time_param = '%sZ' % date.isoformat('_', timespec='seconds')
        #ds_hmi = 'hmi.M_720s[%s]{magnetogram}' % time_param
        #keys_hmi = self.drms_client.keys(ds_hmi)
        #header_hmi, segment_hmi = self.drms_client.query(ds_hmi, key=','.join(keys_hmi), seg='magnetogram')
        #if len(header_hmi) != 1 or np.any(header_hmi.QUALITY != 0):
        #    self.fetchDataFallback(date)
        #    return

        # query EUV
        time_param = '%sZ' % date.isoformat('_', timespec='seconds')
        ds_euv = 'aia.lev1_euv_12s[%s][%s]{image}' % (time_param, ','.join(self.wavelengths))
        keys_euv = self.drms_client.keys(ds_euv)
        header_euv, segment_euv = self.drms_client.query(ds_euv, key=','.join(keys_euv), seg='image')
        if len(header_euv) != len(self.wavelengths) or np.any(header_euv.QUALITY != 0):
            self.fetchDataFallback(date)
            return

        queue = []
        #for (idx, h), s in zip(header_hmi.iterrows(), segment_hmi.magnetogram):
        #    queue += [(h.to_dict(), s, date)]
        for (idx, h), s in zip(header_euv.iterrows(), segment_euv.image):
            queue += [(h.to_dict(), s, date)]

        with multiprocessing.Pool(self.n_workers) as p:
            p.map(self.download, queue)
        logging.info('Finished: %s' % id)

    def fetchDataFallback(self, date):
        """
        Download the data for the given date using fallback.

        Args:
            date (datetime): The date for which the data should be downloaded.

        Returns:
            list: List of paths to the downloaded files.
        """
        id = date.isoformat()

        logging.info('Fallback download: %s' % id)
        # query Magnetogram
        t = date - timedelta(hours=24)
        ds_hmi = 'hmi.M_720s[%sZ/12h@720s]{magnetogram}' % t.replace(tzinfo=None).isoformat('_', timespec='seconds')
        keys_hmi = self.drms_client.keys(ds_hmi)
        header_tmp, segment_tmp = self.drms_client.query(ds_hmi, key=','.join(keys_hmi), seg='magnetogram')
        assert len(header_tmp) != 0, 'No data found!'
        date_str = header_tmp['DATE__OBS'].replace('MISSING', '').str.replace('60', '59')  # fix date format
        date_diff = np.abs(pd.to_datetime(date_str).dt.tz_localize(None) - date)
        # sort and filter
        header_tmp['date_diff'] = date_diff
        header_tmp.sort_values('date_diff')
        segment_tmp['date_diff'] = date_diff
        segment_tmp.sort_values('date_diff')
        cond_tmp = header_tmp.QUALITY == 0
        header_tmp = header_tmp[cond_tmp]
        segment_tmp = segment_tmp[cond_tmp]
        assert len(header_tmp) > 0, 'No valid quality flag found'
        # replace invalid
        header_hmi = header_tmp.iloc[0].drop('date_diff')
        segment_hmi = segment_tmp.iloc[0].drop('date_diff')
        ############################################################
        # query EUV
        header_euv, segment_euv = [], []
        t = date - timedelta(hours=6)
        for wl in self.wavelengths:
            euv_ds = 'aia.lev1_euv_12s[%sZ/12h@12s][%s]{image}' % (
                t.replace(tzinfo=None).isoformat('_', timespec='seconds'), wl)
            keys_euv = self.drms_client.keys(euv_ds)
            header_tmp, segment_tmp = self.drms_client.query(euv_ds, key=','.join(keys_euv), seg='image')
            assert len(header_tmp) != 0, 'No data found!'
            date_str = header_tmp['DATE__OBS'].replace('MISSING', '').str.replace('60', '59')  # fix date format
            date_diff = (pd.to_datetime(date_str).dt.tz_localize(None) - date).abs()
            # sort and filter
            header_tmp['date_diff'] = date_diff
            header_tmp.sort_values('date_diff')
            segment_tmp['date_diff'] = date_diff
            segment_tmp.sort_values('date_diff')
            cond_tmp = header_tmp.QUALITY == 0
            header_tmp = header_tmp[cond_tmp]
            segment_tmp = segment_tmp[cond_tmp]
            assert len(header_tmp) > 0, 'No valid quality flag found'
            # replace invalid
            header_euv.append(header_tmp.iloc[0].drop('date_diff'))
            segment_euv.append(segment_tmp.iloc[0].drop('date_diff'))

        queue = []
        #queue += [(header_hmi.to_dict(), segment_hmi.magnetogram, date)]
        for h, s in zip(header_euv, segment_euv):
            queue += [(h.to_dict(), s.image, date)]

        with multiprocessing.Pool(self.n_workers) as p:
            p.map(self.download, queue)

        logging.info('Finished: %s' % id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download SDO data from JSOC with quality check and fallback')
    parser.add_argument('--download_dir', type=str, help='path to the download directory.')
    parser.add_argument('--email', type=str, help='registered email address for JSOC.')
    parser.add_argument('--start_date', type=str, help='start date in format YYYY-MM-DD.')
    parser.add_argument('--end_date', type=str, help='end date in format YYYY-MM-DD.', required=False,
                        default=str(datetime.now()).split(' ')[0])

    args = parser.parse_args()
    download_dir = args.download_dir
    start_date = args.start_date
    end_date = args.end_date
    #download_dir = '/Users/christophschirninger/PycharmProjects/MDRAIT_ITI'

    [os.makedirs(os.path.join(download_dir, str(c)), exist_ok=True) for c in [131, 171, 193, 211, 304, 335]]
    downloader = SDODownloader(base_path=download_dir, email=args.email)
    start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    #end_date = datetime.now()
    end_date_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    for d in [start_date_datetime + i * timedelta(days=1) for i in
              range((end_date_datetime - start_date_datetime) // timedelta(days=1))]:
        downloader.downloadDate(d)
