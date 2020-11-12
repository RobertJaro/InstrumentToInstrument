import glob

from sunpy.cm import cm
from tqdm import tqdm

from iti.data.editor import LoadMapEditor, NormalizeRadiusEditor, secchi_norms

from matplotlib import pyplot as plt

files = glob.glob('/gss/r.jarolim/data/stereo_prep/valid/secchi_171/*.fits')

fig, axs = plt.subplots(len(files), 1, figsize=(1, len(files)))
for ax, f in tqdm(zip(axs, files), total=len(files)):
    s_map, _ = LoadMapEditor().call(f)
    s_map = NormalizeRadiusEditor(1024).call(s_map)
    ax.imshow(s_map.data, norm=secchi_norms[171], cmap=cm.sdoaia171)

plt.savefig('/gss/r.jarolim/data/stereo_prep/valid/images/171.jpg', dpi=300)
plt.close()