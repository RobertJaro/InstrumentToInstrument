import glob
import os
import random
from multiprocessing import Pool

from scipy.stats import median_absolute_deviation
from sunpy.map import all_coordinates_from_map, Map
from tqdm import tqdm

from iti.data.editor import NormalizeRadiusEditor, KSOPrepEditor, LimbDarkeningCorrectionEditor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from matplotlib import pyplot as plt

import numpy as np

from astropy import units as u

files = sorted(glob.glob("/gss/r.jarolim/data/kso_synoptic/*.fts.gz"))

kso_prep = KSOPrepEditor()
noormalize_radius = NormalizeRadiusEditor(1024, 0)
limb_correct = LimbDarkeningCorrectionEditor()

def getValues(f):
    s_map = Map(f)
    #
    s_map = kso_prep.call(s_map)
    s_map = noormalize_radius.call(s_map)
    s_map = limb_correct.call(s_map)

    return s_map.data, np.nanmedian(s_map.data), np.nanstd(s_map.data)


with Pool(8) as p:
    values = [v for v in tqdm(p.imap(getValues, files[::50]), total=len(files[::50]))]

median = np.mean([m for v, m, s in values])
print('Median', median)
std = np.mean([s for v, m, s in values])
print('Std', std)

plt.hist(np.concatenate([np.ravel(v) for v, m, s in values]), bins=100)
plt.savefig('/gss/r.jarolim/data/converted/kso_synoptic_img/hist.jpg')
plt.close()