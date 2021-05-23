import glob
import os
from warnings import simplefilter

import numpy as np
from matplotlib import pyplot as plt
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from iti.data.editor import soho_norms

plot_path = '/gss/r.jarolim/data/converted/soho_iti2021_prep_imgs'
data_path = '/gss/r.jarolim/data/soho_iti2021_prep'
os.makedirs(plot_path, exist_ok=True)

dirs = ['171', '195', '284', '304', 'mag', ]
cmaps = [cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia304, 'gray']

basenames = [[os.path.basename(path) for path in glob.glob('%s/%s/*.fits' % (data_path, dir))] for
             dir in dirs]
basenames = set(basenames[0]).intersection(*basenames)

for basename in tqdm(basenames):
    simplefilter('ignore')
    soho_cube = [Map('%s/%s/%s' % (data_path, dir, basename)) for dir in dirs]
    #
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    [ax.set_axis_off() for ax in np.ravel(axs)]
    for ax, s_map, cmap, norm in zip(axs, soho_cube, cmaps, soho_norms.values()):
        s_map = s_map.rotate(recenter=True)
        s_map.plot(axes=ax, cmap=cmap, norm=norm, title=None)
    plt.tight_layout(0)
    fig.savefig(os.path.join(plot_path, '%s') % basename.replace('.fits', '.jpg'))
    plt.close(fig)
