import datetime
from datetime import timedelta

from dateutil.parser import parse

from iti.download.download_hmi_continuum import HMIContinuumDownloader

#hinode_sample = '/gss/r.jarolim/data/hinode/level1/FG20131118_174620.3.fits'
hinode_date = datetime.datetime(2014, 11, 22, 1, 30)#parse(hinode_sample[-22:-7].replace('_', 'T'))
hinode_dates = [hinode_date - timedelta(seconds=45) * i for i in range(2 * 80)]

fetcher = HMIContinuumDownloader(ds_path='/gss/r.jarolim/data/hmi_video_2014_11_22', num_worker_threads=2, ignore_quality=True,
                                 series='hmi.Ic_45s')
fetcher.fetchDates(hinode_dates)
