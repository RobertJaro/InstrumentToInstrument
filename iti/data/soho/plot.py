import glob
import os
import shutil
from multiprocessing import Pool
from warnings import simplefilter

from matplotlib import pyplot as plt
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from iti.data.editor import soho_norms

plot_path = '/gpfs/gpfs0/robert.jarolim/data/iti/soho_imgs'
data_path = '/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep/304'
os.makedirs(plot_path, exist_ok=True)

files = sorted(glob.glob(os.path.join(data_path, '*.fits')))
cmap = cm.sdoaia304
norm = soho_norms[304]


def plot(file):
    simplefilter('ignore')
    s_map = Map(file)
    #
    plt.figure(figsize=(2, 2))
    plt.imshow(s_map.data, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(plot_path, os.path.basename(file).replace('fits', 'jpg')), dpi=100)
    plt.close()


with Pool(2) as p:
    [None for _ in tqdm(p.imap_unordered(plot, files), total=len(files))]

shutil.make_archive('/gpfs/gpfs0/robert.jarolim/data/iti/soho_imgs', 'zip', '/gpfs/gpfs0/robert.jarolim/data/iti/soho_imgs')

to_remove = [
    '1997-07-25T02:20:19.fits',
    '2005-08-24T01:19:35.fits',
    '2009-06-04T07:19:19.fits',
    '2014-06-17T01:21:20.fits',
    '2020-11-01T01:19:42.fits',
]

wls = [171, 195, 211, 304]
for bn in to_remove:
    for wl in wls:
        f = os.path.join('/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep/%s/%s' % (wl, bn))
        if os.path.exists(f):
            print(f)
            os.remove(f)
    f = os.path.join('/gpfs/gpfs0/robert.jarolim/data/converted/soho_1024/%s' % (bn.replace('fits', 'npy')))
    if os.path.exists(f):
        print(f)
        os.remove(f)