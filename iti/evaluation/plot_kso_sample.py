import os

from sunpy.map import Map

from iti.data.dataset import KSOFlatDataset



from matplotlib import pyplot as plt

import numpy as np

base_path = "/gss/r.jarolim/iti/film_v8"
prediction_path = os.path.join(base_path, 'translation')
os.makedirs(prediction_path, exist_ok=True)

# load maps
ccd_dataset = KSOFlatDataset("/gss/r.jarolim/data/kso_synoptic", 1024)

# translate
s_map, kso_img = Map(ccd_dataset.data[1200]), ccd_dataset[1200]

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.set_axis_off()
ax.imshow(np.array(kso_img)[0], cmap='gray', vmin=-1, vmax=1)
plt.tight_layout(0)
fig.savefig(os.path.join(prediction_path, '%s_kso.jpg' % s_map.date.datetime.isoformat('T')), dpi=300)
plt.close(fig)
