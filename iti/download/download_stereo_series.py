from multiprocessing import Pool

import drms
import logging
from sunpy.net import Fido, attrs as a
from tqdm import tqdm

from iti.download.download_stereo import STEREODownloader
from astropy import units as u

if __name__ == '__main__':
    base_path = '/localdata/USER/rja/stereo_iti2021_series'  # sys.argv[1]
    n_worker = 4  # int(sys.argv[2])

    drms_client = drms.Client(email='robert.jarolim@uni-graz.at', verbose=False)
    download_util = STEREODownloader(base_path)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ])

    search = Fido.search(a.Time('2012-11-01', '2012-12-01'),
                         a.Provider('SSC'), a.Instrument('SECCHI'), a.Wavelength(171 * u.AA), a.Source('STEREO_A'))
    dates = search['vso']['Start Time']
    sources = search['vso']['Source']
    logging.info("TOTAL DATES: %d" % (len(dates)))
    dates = [d.datetime for d in dates]

    with Pool(n_worker) as p:
        [None for _ in tqdm(p.imap_unordered(download_util.downloadDate, zip(dates, sources)), total=len(dates))]