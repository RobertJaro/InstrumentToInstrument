import glob
import os
from warnings import simplefilter

from dateutil.parser import parse
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from iti.prediction.translate import SOHOToSDO

# init
base_path = "/gss/r.jarolim/iti/soho_sdo_v24"
prediction_path = '/gss/r.jarolim/data/soho_iti_converted'
os.makedirs(prediction_path, exist_ok=True)
# check existing
existing = [os.path.basename(f) for f in glob.glob(os.path.join(prediction_path, 'mag', '*.fits'))]
# create translator
basenames_soho = [[os.path.basename(f) for f in glob.glob('/gss/r.jarolim/data/soho_iti2021_prep/%s/*.fits' % wl)] for
                  wl in ['171', '195', '284', '304', 'mag']]
basenames_soho = set(basenames_soho[0]).intersection(*basenames_soho[1:])
basenames_soho = sorted(list(basenames_soho))
basenames_soho = [b for b in basenames_soho if b not in existing]
dates_soho = sorted([parse(f.split('.')[0]) for f in basenames_soho])

translator = SOHOToSDO(model_path=os.path.join(base_path, 'generator_AB.pt'))

# translate
dirs = ['171', '195', '284', '304', 'mag', ]
[os.makedirs(os.path.join(prediction_path, d), exist_ok=True) for d in dirs]

iti_maps = translator.translate('/gss/r.jarolim/data/soho_iti2021_prep', basenames=basenames_soho)
for iti_cube, date in tqdm(zip(iti_maps, dates_soho)):
    simplefilter('ignore')  # ignore int conversion warning
    for s_map, d in zip(iti_cube, dirs):
        path = os.path.join(os.path.join(prediction_path, d, '%s.fits') % date.isoformat('T'))
        if os.path.exists(path):
            os.remove(path)
        s_map.save(path)
