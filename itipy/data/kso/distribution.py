import glob
import os
import random
from multiprocessing import Pool

from scipy.stats import median_absolute_deviation
from sunpy.map import all_coordinates_from_map, Map
from tqdm import tqdm

from itipy.data.editor import NormalizeRadiusEditor, KSOPrepEditor, LimbDarkeningCorrectionEditor, KSOFilmPrepEditor, \
    LoadFITSEditor



from matplotlib import pyplot as plt

import numpy as np

from astropy import units as u

files = sorted(glob.glob("/gss/r.jarolim/data/filtered_kso_plate/*.fts.gz"))

load_fits = LoadFITSEditor()
kso_prep = KSOFilmPrepEditor()
noormalize_radius = NormalizeRadiusEditor(1024, 0)
limb_correct = LimbDarkeningCorrectionEditor()

def getValues(f):
    data, kwargs = load_fits.call(f)
    #
    s_map = kso_prep.call(data, **kwargs)
    s_map = noormalize_radius.call(s_map)
    s_map = limb_correct.call(s_map)

    return s_map.data


with Pool(8) as p:
    values = [v for v in tqdm(p.imap(getValues, files[::10]), total=len(files[::10]))]

print('Average Max Value:', np.mean([np.nanmax(v) for v in values]))
print('Average Min Value:', np.mean([np.nanmin(v) for v in values]))
plt.hist(np.ravel(values), bins=100)
plt.semilogy()
plt.savefig('/gss/r.jarolim/data/converted/kso_film_img/hist.jpg')
plt.close()