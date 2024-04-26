import gc
import glob
import os
import shutil
from warnings import simplefilter

from astropy import units as u
from dateutil.parser import parse
from tqdm import tqdm

from iti.translate import STEREOToSDO

# init
base_path = "/gpfs/gpfs0/robert.jarolim/iti/stereo_to_sdo_v1"
prediction_path = '/gpfs/gpfs0/robert.jarolim/data/iti/stereo_synchronic_iti_converted'
data_path = '/gpfs/gpfs0/robert.jarolim/data/iti/stereo_synchronic_prep'
os.makedirs(prediction_path, exist_ok=True)

# check existing
existing = [os.path.basename(f) for f in glob.glob(os.path.join(prediction_path, '171', '*.fits'))]

# create translator
basenames_stereo = [[os.path.basename(f) for f in glob.glob('%s/%s/*.fits' % (data_path, wl))] for
                  wl in ['171', '195', '284', '304']]
basenames_stereo = set(basenames_stereo[0]).intersection(*basenames_stereo[1:])
basenames_stereo = sorted(list(basenames_stereo))
basenames_stereo = [b for b in basenames_stereo if b not in existing]

translator = STEREOToSDO(model_path=os.path.join(base_path, 'generator_AB.pt'))

# translate
dirs = ['171', '195', '284', '304', ]
[os.makedirs(os.path.join(prediction_path, d), exist_ok=True) for d in dirs]

iti_maps = translator.translate(data_path, basenames=basenames_stereo)
for iti_cube, bn in tqdm(zip(iti_maps, basenames_stereo), total=len(basenames_stereo)):
    simplefilter('ignore')  # ignore int conversion warning
    for s_map, d in zip(iti_cube, dirs):
        path = os.path.join(os.path.join(prediction_path, d, bn))
        if os.path.exists(path):
            os.remove(path)
        s_map.resample((512, 512) * u.pix).save(path)
        gc.collect()
