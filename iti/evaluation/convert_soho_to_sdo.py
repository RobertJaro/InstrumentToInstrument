import gc
import glob
import os
from warnings import simplefilter

from astropy import units as u
from dateutil.parser import parse
from tqdm import tqdm

from iti.translate import SOHOToSDO

# init
base_path = "/gpfs/gpfs0/robert.jarolim/iti/soho_sdo_v1"
prediction_path = '/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti_converted'
data_path = '/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep'
os.makedirs(prediction_path, exist_ok=True)

# check existing
existing = [os.path.basename(f) for f in glob.glob(os.path.join(prediction_path, '171', '*.fits'))]
# create translator
basenames_soho = [[os.path.basename(f) for f in glob.glob('%s/%s/*.fits' % (data_path, wl))] for
                  wl in ['171', '195', '284', '304', 'mag']]
basenames_soho = set(basenames_soho[0]).intersection(*basenames_soho[1:])
basenames_soho = sorted(list(basenames_soho))
basenames_soho = [b for b in basenames_soho if b not in existing]
dates_soho = [parse(f.split('.')[0]) for f in basenames_soho]

translator = SOHOToSDO(model_path=os.path.join(base_path, 'generator_AB.pt'))

# translate
dirs = ['171', '195', '284', '304', 'mag']
[os.makedirs(os.path.join(prediction_path, d), exist_ok=True) for d in dirs]

iti_maps = translator.translate(data_path, basenames=basenames_soho)
for iti_cube, bn in tqdm(zip(iti_maps, basenames_soho), total=len(basenames_soho)):
    simplefilter('ignore')  # ignore int conversion warning
    for s_map, d in zip(iti_cube, dirs):
        path = os.path.join(os.path.join(prediction_path, d, bn))
        if os.path.exists(path):
            os.remove(path)
        s_map.resample((512, 512) * u.pix).save(path)
        gc.collect()
