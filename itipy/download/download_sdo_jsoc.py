import argparse
import glob
import os
import shutil
from datetime import timedelta, datetime

import drms
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

parser = argparse.ArgumentParser(description='Download SDO data from JSOC')
parser.add_argument('--download_dir', type=str, help='path to the download directory.')
parser.add_argument('--email', type=str, help='registered email address for JSOC.')
parser.add_argument('--channels', '-channels', nargs='+', required=False,
                    default=['171', '193', '211', '304', '6173'],
                    help='subset of channels to load. The order must match the input channels of the model.')


args = parser.parse_args()
download_dir = args.download_dir
channels = args.channels
euv_channels = [c for c in channels if c != '6173']

[os.makedirs(os.path.join(download_dir, str(c)), exist_ok=True) for c in channels]

client = drms.Client(verbose=True, email=args.email)


def round_date(t):
    if t.minute >= 30:
        return t.replace(second=0, microsecond=0, minute=0) + timedelta(hours=1)
    else:
        return t.replace(second=0, microsecond=0, minute=0)


def download(ds):
    r = client.export(ds, method='url-tar', protocol='fits')
    r.wait()
    download_result = r.download(download_dir)
    for f in download_result.download:
        shutil.unpack_archive(f, os.path.join(download_dir))
        os.remove(f)
    for f in glob.glob(os.path.join(download_dir, '*.fits')):
        f_info = os.path.basename(f).split('.')
        channel = f_info[3]
        if f_info[0] == 'hmi':
            channel = '6173'
            date = round_date(parse(f_info[2][:-4].replace('_', 'T')))
        else:
            date = round_date(parse(f_info[2][:-1]))
        shutil.move(f, os.path.join(download_dir, str(channel), date.isoformat('T', timespec='hours') + '.fits'))
    [os.remove(f) for f in glob.glob(os.path.join(download_dir, '*.*'))]


def download_month(year, month):
    tstart = datetime(year, month, 1, 0, 0, 0)
    tend = tstart + relativedelta(months=1) - timedelta(hours=6)
    download_date_range(tstart, tend)


def download_date_range(tstart, tend):
    tstart, tend = tstart.isoformat('_', timespec='seconds'), tend.isoformat('_', timespec='seconds')
    print('Download AIA: %s -- %s' % (tstart, tend))
    download('aia.lev1_euv_12s[%sZ-%sZ@6h][%s]{image}' % (tstart, tend, ','.join(euv_channels)), )
    if '6173' in channels:
        print('Download HMI: %s -- %s' % (tstart, tend))
        download('hmi.M_720s[%sZ-%sZ@6h]{magnetogram}' % (tstart, tend), )


tstart = datetime(2010, 5, 1)
tend = datetime.now()
td = timedelta(days=30)

dates = [tstart + i * td for i in range((tend - tstart) // td)]
for d in dates:
    download_date_range(d, d + td)
