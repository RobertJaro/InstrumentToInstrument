import glob

import os
from datetime import datetime

from dateutil.parser import parse
from tqdm import tqdm



from itipy.translate import KSOLowToHigh

from matplotlib import pyplot as plt
import numpy as np

# init
base_path = '/gss/r.jarolim/iti/kso_quality_1024_v5'
prediction_path = os.path.join(base_path, 'series')
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = KSOLowToHigh(resolution=1024, model_path=os.path.join(base_path, 'generator_AB.pt'))


# load maps
map_files = np.array(glob.glob('/gss/r.jarolim/data/kso_full_days_converted/kso_full_days_converted/*.fits', recursive=True))
dates = np.array([parse(os.path.basename(f).split('.')[0]) for f in map_files])

map_files = map_files[dates > datetime(2019, 1, 26, 13)]
dates = dates[dates > datetime(2019, 1, 26, 13)]

# translate
for (_, kso_img, iti_img), date in tqdm(zip(translator.translate(map_files), dates)):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    [ax.set_axis_off() for ax in axs]
    axs[0].imshow(np.array(kso_img)[0], cmap='gray', vmin=-1, vmax=1)
    axs[1].imshow(iti_img[0], cmap='gray', vmin=-1, vmax=1)
    plt.tight_layout()
    fig.savefig(os.path.join(prediction_path, '%s.jpg' % date.isoformat()), dpi=300)
    plt.close(fig)

