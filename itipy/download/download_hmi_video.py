import argparse
import datetime
from datetime import timedelta

from itipy.download.download_hmi_continuum import HMIContinuumDownloader

parser = argparse.ArgumentParser(description='Download SDO/HMI data for video')
parser.add_argument('--download_dir', type=str, help='path to the download directory.')
parser.add_argument('--email', type=str, help='registered email address for JSOC.')
args = parser.parse_args()

hinode_date = datetime.datetime(2014, 11, 22, 1, 30)
hinode_dates = [hinode_date - timedelta(seconds=45) * i for i in range(2 * 80)]

fetcher = HMIContinuumDownloader(ds_path=args.download_dir, num_worker_threads=2,
                                 ignore_quality=True,
                                 series='hmi.Ic_45s', email=args.email)
fetcher.fetchDates(hinode_dates)
