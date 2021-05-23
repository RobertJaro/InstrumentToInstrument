import glob

import os

from iti.data.dataset import KSOFlatDataset

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from sunpy.map import Map

from iti.prediction.translate import KSOLowToHigh, KSOFlatConverter

from matplotlib import pyplot as plt
import numpy as np

# init
base_path = '/gss/r.jarolim/iti/kso_quality_1024_v5'
prediction_path = os.path.join(base_path, 'translation')
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = KSOLowToHigh(resolution=1024, model_path=os.path.join(base_path, 'generator_AB.pt'))
converter = KSOFlatConverter(1024)


# load maps
map_files = list(glob.glob('/gss/r.jarolim/data/kso_comparison_iti2021/**/*.fts.gz', recursive=True))
lq_files = sorted([f for f in map_files if 'ref_' not in os.path.basename(f)])
hq_files = sorted([f for f in map_files if 'ref_' in os.path.basename(f)])
dates = [Map(f).date.datetime for f in lq_files]
ref_imgs, metas = converter.convert(hq_files)


def saveimg(img, path):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.set_axis_off()
    ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.tight_layout(0)
    fig.savefig(path, dpi=300)
    plt.close(fig)


# translate
for (_, kso_img, iti_img), ref_img, date in zip(translator.translate(lq_files), ref_imgs, dates):
    saveimg(np.array(kso_img)[0], os.path.join(prediction_path, '%s_kso.jpg' % date.isoformat()))
    saveimg(iti_img[0], os.path.join(prediction_path, '%s_iti.jpg' % date.isoformat()))
    saveimg(ref_img, os.path.join(prediction_path, '%s_ref.jpg' % date.isoformat()))

