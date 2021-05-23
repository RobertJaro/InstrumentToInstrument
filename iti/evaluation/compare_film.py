import glob
import os

from dateutil.parser import parse

from iti.prediction.translate import KSOFilmToCCD

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from matplotlib import pyplot as plt

import numpy as np

base_path = "/gss/r.jarolim/iti/film_v7"
prediction_path = os.path.join(base_path, 'translation')
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = KSOFilmToCCD(resolution=512, model_path=os.path.join(base_path, 'generator_AB.pt'))

# load maps
map_files = sorted(list(glob.glob('/gss/r.jarolim/data/filtered_kso_plate/*.fts.gz', recursive=True)))

# translate
for s_map, kso_img, iti_img in translator.translate(map_files):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.set_axis_off()
    ax.imshow(np.array(kso_img)[0], cmap='gray', vmin=-1, vmax=1)
    plt.tight_layout(0)
    fig.savefig(os.path.join(prediction_path, '%s_film.jpg' % s_map.date.datetime.isoformat('T')), dpi=300)
    plt.close(fig)
    #
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.set_axis_off()
    ax.imshow(np.array(iti_img)[0], cmap='gray', vmin=-1, vmax=1)
    plt.tight_layout(0)
    fig.savefig(os.path.join(prediction_path, '%s_reconstruction.jpg' % s_map.date.datetime.isoformat('T')), dpi=300)
    plt.close(fig)
