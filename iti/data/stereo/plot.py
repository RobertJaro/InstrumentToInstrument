import glob
import os
from multiprocessing import Pool
from warnings import simplefilter

import numpy as np
from matplotlib import pyplot as plt
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from iti.data.editor import stereo_norms

plot_path = '/gss/r.jarolim/data/converted/stereo_iti2021_prep_imgs'
data_path = '/gss/r.jarolim/data/stereo_iti2021_prep'
os.makedirs(plot_path, exist_ok=True)

dirs = ['171', '195', '284', '304', ]
cmaps = [cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia304]

basenames = [[os.path.basename(path) for path in glob.glob('%s/%s/*.fits' % (data_path, dir))] for
             dir in dirs]
basenames = set(basenames[0]).intersection(*basenames)


def plotBasename(basename):
    simplefilter('ignore')
    stereo_cube = [Map('%s/%s/%s' % (data_path, dir, basename)) for dir in dirs]
    #
    fig, axs = plt.subplots(1, len(stereo_cube), figsize=(3 * len(stereo_cube), 3))
    [ax.set_axis_off() for ax in np.ravel(axs)]
    for ax, s_map, cmap, norm in zip(axs, stereo_cube, cmaps, stereo_norms.values()):
        s_map = s_map.rotate(recenter=True)
        s_map.plot(axes=ax, cmap=cmap, norm=norm, title=None)
    plt.tight_layout(0)
    fig.savefig(os.path.join(plot_path, '%s') % basename.replace('.fits', '.jpg'))
    plt.close(fig)

with Pool(8) as p:
    [None for _ in tqdm(p.imap(plotBasename, basenames), total=len(basenames))]
