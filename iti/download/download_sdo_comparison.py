import glob
import os
from datetime import timedelta, datetime

from dateutil.parser import parse
from tqdm import tqdm

from iti.download.download_sdo import SDODownloader

downloader = SDODownloader(base_path="/gss/r.jarolim/data/sdo_comparison")
basenames_soho = [[os.path.basename(f) for f in glob.glob('/gss/r.jarolim/data/soho_iti2021_prep/%s/*.fits' % wl)]
                  for wl in ['171', '195', '284', '304', 'mag']]
basenames_soho = set(basenames_soho[0]).intersection(*basenames_soho[1:])
dates = sorted([parse(f.split('.')[0]) for f in basenames_soho])
dates = [d for d in dates if d > datetime(2010, 5, 12)]
for d in tqdm(dates):
    try:
        downloader.downloadDate(d)
    except Exception as ex:
        print(ex)
