import argparse
import logging
import os
import shutil
from datetime import timedelta, datetime
from multiprocessing import Pool
from urllib.request import urlopen
from warnings import simplefilter

import drms
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io.fits import getheader, HDUList
from dateutil.relativedelta import relativedelta
from sunpy.map import Map
from sunpy.net import Fido, attrs as a
from tqdm import tqdm


class SOHODownloader:
    """
    Class to download SOHO EIT and MDI data from the VSO.

    Args:
        base_path (str): Path to the directory where the downloaded data should be stored.
    """
    def __init__(self, base_path, wavelengths=['171', '195', '284', '304']):
        self.base_path = base_path
        self.wavelengths = [str(wl) for wl in wavelengths]
        [os.makedirs(os.path.join(base_path, wl), exist_ok=True) for wl in self.wavelengths]

    def downloadDate(self, date):
        """
        Download the data for the given date.

        Args:
            date (datetime): The date for which the data should be downloaded.

        Returns:
            list: List of paths to the downloaded files.
        """
        files = []
        try:
            # Download EIT
            for wl in self.wavelengths:
                files += [self.downloadEIT(date, wl)]
            # Download MDI
            if date < datetime(2011, 4, 12):  # No MDI data past 2011
                files += [self.downloadMDI(date)]
            logging.info('Download complete %s' % date.isoformat())
        except Exception as ex:
            logging.error('Unable to download %s: %s' % (date.isoformat(), str(ex)))
            [os.remove(f) for f in files]

    def downloadEIT(self, query_date, wl):
        """
        Download the EIT data for the given date and wavelength.

        Args:
            query_date (datetime): The date for which the data should be downloaded.
            wl (int): The wavelength of the data.

        Returns:
            str: Path to the downloaded file.
        """
        file_path = os.path.join(self.base_path, str(wl), "%s.fits" % query_date.isoformat("T", timespec='seconds'))
        if os.path.exists(file_path):
            return file_path  # skip existing downloads (e.g. retry)
        #
        search = Fido.search(a.Time(query_date - timedelta(minutes=30), query_date + timedelta(minutes=30)),
                             a.Instrument('EIT'), a.Wavelength(wl * u.AA), a.Provider("SDAC"))
        assert search.file_num > 0, "No data found for %s (%s)" % (query_date.isoformat(), wl)
        search = sorted(search['vso'], key=lambda x: abs(x['Start Time'].datetime - query_date).total_seconds())
        for entry in search:
            files = Fido.fetch(entry, path=self.base_path, progress=False)
            if len(files) != 1:
                continue
            file = files[0]
            header = getheader(file)
            if header['NAXIS1'] != 1024 or header['NAXIS2'] != 1024 or 'N_MISSING_BLOCKS =    0' not in \
                    header['COMMENT'][-1]:
                os.remove(file)
                continue
            shutil.move(file, file_path)
            return file_path

        raise Exception("No valid file found for %s (%s)!" % (query_date.isoformat(), wl))

    def downloadMDI(self, download_date):
        """
        Download the MDI data for the given date.

        Args:
            download_date (datetime): The date for which the data should be downloaded.

        Returns:
            str: Path to the downloaded file.
        """
        simplefilter('ignore')
        file_path = os.path.join(self.base_path, self.dirs[0],
                                 "%s.fits" % download_date.isoformat("T", timespec='seconds'))
        if os.path.exists(file_path):
            return file_path
        time_param = '%sZ/192m' % (download_date - timedelta(minutes=96)).isoformat('_', timespec='seconds')
        ds_mdi = 'mdi.fd_M_96m_lev182[%s]{data}' % time_param
        keys_mdi = drms_client.keys(ds_mdi)
        headers = drms_client.query(ds_mdi, key=','.join(keys_mdi))
        headers['url'] = drms_client.export(ds_mdi).urls.url
        headers = headers[(headers['DATE__OBS'] != 'MISSING')]
        # ((headers['QUALITY'] == 0) | (headers['QUALITY'] == 512)) &
        headers['time_diff'] = (pd.to_datetime(headers['DATE__OBS']).dt.tz_localize(None) - download_date).abs()
        headers = headers.sort_values('time_diff')
        for i, header in headers.iterrows():
            url_request = urlopen(header.url)
            fits_data = url_request.read()
            hdul = HDUList.fromstring(fits_data)
            hdul.verify('silentfix')
            data = hdul[1].data
            header_dict = header.to_dict()
            header_dict['DATE_OBS'] = header_dict['DATE__OBS']
            del header_dict['time_diff']
            del header_dict['url']
            s_map = Map(data, header_dict)
            s_map.save(file_path)
            return file_path
        raise Exception('No valid magnetogram found!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download SOHO data')
    parser.add_argument('--download_dir', type=str, help='path to the download directory.')
    parser.add_argument('--n_workers', type=str, help='number of parallel threads.', required=False, default=4)
    parser.add_argument('--start_date', type=str, help='start date for the download.', required=False)
    parser.add_argument('--end_date', type=str, help='end date for the download.', required=False, default=str(datetime.now()).split(' ')[0])

    args = parser.parse_args()
    base_path = args.download_dir
    n_workers = args.n_workers
    start_date = args.start_date
    end_date = args.end_date

    drms_client = drms.Client(email='robert.jarolim@uni-graz.at', verbose=False)
    download_util = SOHODownloader(base_path)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ])
    start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    num_months = (end_date_datetime.year - start_date_datetime.year) * 12 + (end_date_datetime.month - start_date_datetime.month)
    month_dates = [start_date_datetime + i * relativedelta(months=1) for i in range(num_months)]
    for date in month_dates:
            search = Fido.search(a.Time(date, date + relativedelta(months=1)),
                                 a.Provider("SDAC"), a.Instrument('EIT'),
                                 a.Wavelength(304 * u.AA))
            if search.file_num == 0:
                continue
            dates = search['vso']['Start Time']
            logging.info("TOTAL DATES (%s): %d" % (date.isoformat(), len(dates)))
            step = int(np.floor(len(dates) / 60)) if len(dates) > 60 else 1
            dates = [d.datetime for d in dates[::step]]

            with Pool(n_workers) as p:
                [None for _ in tqdm(p.imap_unordered(download_util.downloadDate, dates), total=len(dates))]
