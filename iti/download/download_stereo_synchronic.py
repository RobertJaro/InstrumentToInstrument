import argparse
import datetime
import os
import shutil

import astropy.units as u
from astropy.io import fits
from dateutil.parser import parse
from sunpy.net import Fido
from sunpy.net import attrs as a
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Download aligned STEREO A+B data')
parser.add_argument('--download_dir', type=str, help='path to the download directory.')

args = parser.parse_args()
download_dir = args.download_dir

wavelengths = [171, 195, 284, 304, ]
t_start = datetime.datetime(2016, 4, 1)
t_end = datetime.datetime.now()
td = datetime.timedelta(days=30)

times = [t_start + i * td for i in range((t_end - t_start) // td)]
[os.makedirs(os.path.join(download_dir, str(wl)), exist_ok=True) for wl in wavelengths]

def round_hour(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
            + datetime.timedelta(hours=t.minute // 30))

for time in tqdm(times):
    queries = []
    for wl in wavelengths:
        stereo_a = Fido.search(a.Instrument("EUVI"), a.Source('STEREO_A'), a.Time(time, time + td), a.Sample(1 * u.day),
                               a.Wavelength(wl * u.AA))
        if time.year > 2014: # loss of STEREO B
            queries += [stereo_a]
            continue
        stereo_b = Fido.search(a.Instrument("EUVI"), a.Source('STEREO_B'), a.Time(time, time + td), a.Sample(1 * u.day),
                               a.Wavelength(wl * u.AA))
        queries += [stereo_a, stereo_b]

    files = Fido.fetch(*queries, path=download_dir, progress=False)
    for file in files:
        if not os.path.exists(file):
            continue
        header = fits.getheader(file)
        if header['NAXIS1'] < 2048 or header['NAXIS2'] < 2048 or header['NMISSING'] != 0:
            os.remove(file)
            print('invalid file:', file, header['NAXIS1'], header['NAXIS2'], header['NMISSING'])
            continue
        wl = int(header['wavelnth'])
        s_id = 'A' if header['obsrvtry'] == 'STEREO_A' else 'B'
        if 'date-obs' not in header:
            header['date-obs'] = header['date_obs']
        d = round_hour(parse(header['date-obs']))
        target_path = os.path.join(download_dir, str(wl),
                                   '%s_%s.fits' % (d.isoformat('T'), s_id))
        print(file, target_path)
        shutil.move(file, target_path)
