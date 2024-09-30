import argparse
import glob
import os
from datetime import datetime

from dateutil.parser import parse
from sunpy.map import Map
from tqdm import tqdm

from itipy.download.download_sdo import SDODownloader

parser = argparse.ArgumentParser(description='Download SDO data aligned with SOHO observations')
parser.add_argument('--download_dir', type=str, help='path to the download directory.')
parser.add_argument('--soho_path', type=str, help='path to the reference soho data.')
parser.add_argument('--email', type=str, help='registered email address for JSOC.')

args = parser.parse_args()

downloader = SDODownloader(base_path=args.download_dir, email=args.email)
basenames_soho = [
    [os.path.basename(f) for f in glob.glob('%s/%s/*.fits' % (args.soho_path, wl))]
    for wl in ['171', '195', '284', '304', 'mag']]
basenames_soho = set(basenames_soho[0]).intersection(*basenames_soho[1:])
basenames_soho = [f for f in basenames_soho if parse(f.split('.')[0]) > datetime(2010, 5, 12)]  # coarse filter
dates = sorted(
    [Map('%s/mag/%s' % (args.soho_path, f)).date.datetime for f in basenames_soho])
dates = [d for d in dates if d > datetime(2010, 5, 12)]
for d in tqdm(dates):
    try:
        downloader.downloadDate(d)
    except Exception as ex:
        print(ex)
