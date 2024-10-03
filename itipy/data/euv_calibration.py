import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from dateutil.parser import parse
from tqdm import tqdm

from itipy.data.editor import LoadMapEditor, NormalizeRadiusEditor, MapToDataEditor, EITCheckEditor, RemoveOffLimbEditor, \
    AIAPrepEditor, SECCHIPrepEditor

parser = argparse.ArgumentParser(description='Estimate the mean and std of SOHO, STEREO and SDO for calibration.')
parser.add_argument('--soho_path', type=str, help='the path to the soho files.')
parser.add_argument('--stereo_path', type=str, help='the path to the stereo files.')
parser.add_argument('--sdo_path', type=str, help='the path to the sdo files.')
parser.add_argument('--evaluation_path', type=str, help='the path for printing the results.')

args = parser.parse_args()

soho_path = args.soho_path
sdo_path = args.sdo_path
stereo_path = args.stereo_path
evaluation_path = args.evaluation_path

soho_channels = ['171', '195', '284', '304']
sdo_channels = ['171', '193', '211', '304']

eit_files = [sorted(glob.glob(os.path.join(soho_path, c, '*.fits'))) for c in soho_channels]
secchi_files = [sorted(glob.glob(os.path.join(stereo_path, c, '*.fits'))) for c in soho_channels]
aia_files = [sorted(glob.glob(os.path.join(sdo_path, c, '*.fits'))) for c in sdo_channels]


def getEITData(f):
    s_map, _ = LoadMapEditor().call(f)
    s_map = EITCheckEditor().call(s_map)
    s_map = NormalizeRadiusEditor(1024).call(s_map)
    s_map = RemoveOffLimbEditor(fill_value=np.nan).call(s_map)
    data, _ = MapToDataEditor().call(s_map)
    return data


def getAIAData(f):
    s_map, _ = LoadMapEditor().call(f)
    s_map = NormalizeRadiusEditor(1024).call(s_map)
    s_map = AIAPrepEditor(calibration='auto').call(s_map)
    s_map = RemoveOffLimbEditor(fill_value=np.nan).call(s_map)
    data, _ = MapToDataEditor().call(s_map)
    return data


def getSECCHIData(f):
    s_map, _ = LoadMapEditor().call(f)
    deg = [-9.42497209e-05, 2.27153104e+00] if s_map.wavelength.value == 304 else None
    s_map = SECCHIPrepEditor(degradation=deg).call(s_map)
    s_map = NormalizeRadiusEditor(1024).call(s_map)
    s_map = RemoveOffLimbEditor(fill_value=np.nan).call(s_map)
    data, _ = MapToDataEditor().call(s_map)
    return data


def filter_files(files, years=None):
    dates = [parse(os.path.basename(f).split('.')[0]) for f in files]
    files, dates = zip(*[(f, d) for f, d in zip(files, dates) if d.month in list(range(2, 10))])
    if years:
        files, dates = zip(*[(f, d) for f, d in zip(files, dates) if d.year in years])
    df = pd.DataFrame({'date': dates, 'file': files})
    df = df.set_index('date').groupby(pd.Grouper(freq='10D')).first()
    return df[~pd.isna(df['file'])]['file'].tolist()


eit_hist = {}
for c, c_files in zip(soho_channels, eit_files):
    c_files = filter_files(c_files, years=list(range(1996, 2010)))
    with Pool(12) as p:
        data = [np.ravel(m) for m in tqdm(p.imap_unordered(getEITData, c_files), total=len(c_files))]
        data = np.concatenate(data)
    threshold = np.nanmedian(data) + np.nanstd(data)
    data[data > threshold] = np.nan
    eit_hist[c] = [np.nanmean(data), np.nanstd(data)]

secchi_hist = {}
for c, c_files in zip(soho_channels, secchi_files):
    c_files = c_files[::len(c_files) // 100]
    with Pool(12) as p:
        data = [np.ravel(m) for m in tqdm(p.imap_unordered(getSECCHIData, c_files), total=len(c_files))]
        data = np.concatenate(data)
    threshold = np.nanmedian(data) + np.nanstd(data)
    data[data > threshold] = np.nan
    secchi_hist[c] = [np.nanmean(data), np.nanstd(data)]

aia_hist = {}
for c, c_files in zip(sdo_channels, aia_files):
    c_files = c_files[::len(c_files) // 100]
    with Pool(12) as p:
        data = [np.ravel(m) for m in tqdm(p.imap_unordered(getAIAData, c_files), total=len(c_files))]
        data = np.concatenate(data)
    threshold = np.nanmedian(data) + np.nanstd(data)
    data[data > threshold] = np.nan
    aia_hist[c] = [np.nanmean(data), np.nanstd(data)]

with open(os.path.join(evaluation_path, 'calibration.txt'), 'w') as f:
    print('EIT', file=f)
    print(eit_hist, file=f)
    print('SECCHI', file=f)
    print(secchi_hist, file=f)
    print('AIA', file=f)
    print(aia_hist, file=f)
