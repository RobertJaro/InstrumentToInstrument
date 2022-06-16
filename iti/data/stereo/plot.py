import glob
import os
from multiprocessing import Pool
from warnings import simplefilter

from matplotlib import pyplot as plt
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from iti.data.editor import stereo_norms

plot_path = '/gpfs/gpfs0/robert.jarolim/data/iti/stereo_imgs'
data_path = '/gpfs/gpfs0/robert.jarolim/data/iti/stereo_iti2021_prep/304'
os.makedirs(plot_path, exist_ok=True)

files = sorted(glob.glob(os.path.join(data_path, '*.fits')))
cmap = cm.sdoaia304
norm = stereo_norms[304]


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


to_remove = [
    '2006-11-20T00:32:02.fits',
    '2007-01-05T17:24:31.fits',
    '2007-02-05T09:35:42.fits',
    '2007-03-03T03:39:03.fits',
    '2007-05-02T11:04:50.fits',
    '2010-10-13T15:16:00.fits',
    '2016-03-28T00:38:30.fits',
    '2019-03-22T00:14:00.fits',]

wls = [171, 195, 211, 304]
for bn in to_remove:
    for wl in wls:
        f = os.path.join('/gpfs/gpfs0/robert.jarolim/data/iti/stereo_iti2021_prep/%s/%s' % (wl, bn))
        if os.path.exists(f):
            print(f)
            os.remove(f)
    f = os.path.join('/gpfs/gpfs0/robert.jarolim/data/converted/stereo_1024/%s' % (bn.replace('fits', 'npy')))
    if os.path.exists(f):
        print(f)
        os.remove(f)