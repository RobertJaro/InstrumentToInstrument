import os
from random import randint

from matplotlib import pyplot as plt
from sunpy.visualization.colormaps import cm

from iti.data.dataset import HMIContinuumDataset, HinodeDataset, SDODataset, SOHODataset, STEREODataset, KSODataset, \
    KSOFlatDataset, KSOFilmDataset
from iti.data.editor import RandomPatchEditor

base_path = '/gss/r.jarolim/iti/datasets'

sdo_dataset = SDODataset("/gss/r.jarolim/data/ch_detection")
soho_dataset = SOHODataset("/gss/r.jarolim/data/soho/train")

hmi_dataset = HMIContinuumDataset("/gss/r.jarolim/data/hmi_continuum/6173")
hinode_dataset = HinodeDataset('/gss/r.jarolim/data/hinode/level1')
hinode_dataset.addEditor(RandomPatchEditor((640, 640)))

sdo_dataset_fullres = SDODataset("/gss/r.jarolim/data/ch_detection", resolution=4096)
stereo_dataset = STEREODataset("/gss/r.jarolim/data/stereo_prep/train")

q1_dataset = KSODataset("/gss/r.jarolim/data/anomaly_data_set/quality1", 512)
q2_dataset = KSODataset("/gss/r.jarolim/data/kso_general/quality2", 512)

ccd_dataset = KSOFlatDataset("/gss/r.jarolim/data/kso_general/quality1", 512)
film_dataset = KSOFilmDataset("/gss/r.jarolim/data/filtered_kso_plate", 512)

datasets = [soho_dataset, sdo_dataset, hmi_dataset, hinode_dataset, stereo_dataset, sdo_dataset_fullres, q2_dataset,
            q1_dataset, film_dataset, ccd_dataset]


def getSample(d):
    idx = randint(0, len(d))
    return d[idx]


sdo_cmaps = [
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    plt.get_cmap('gray')
]

sdo_names = ['171', '193', '211', '304', 'mag']

for sample, cmap, name in zip(getSample(sdo_dataset), sdo_cmaps, sdo_names):
    plt.figure(figsize=(8,8))
    plt.imshow(sample, vmin=-1, vmax=1, cmap=cmap)
    plt.axis('off')
    plt.tight_layout(0)
    plt.savefig(os.path.join(base_path, 'sdo_%s.jpg' % name), dpi=300)
    plt.close()

for sample, cmap, name in zip(getSample(soho_dataset), sdo_cmaps, sdo_names):
    plt.figure(figsize=(8, 8))
    plt.imshow(sample, vmin=-1, vmax=1, cmap=cmap)
    plt.axis('off')
    plt.tight_layout(0)
    plt.savefig(os.path.join(base_path, 'soho_%s.jpg' % name), dpi=300)
    plt.close()

for sample, cmap, name in zip(getSample(stereo_dataset), sdo_cmaps, sdo_names):
    plt.figure(figsize=(8, 8))
    plt.imshow(sample, vmin=-1, vmax=1, cmap=cmap)
    plt.axis('off')
    plt.tight_layout(0)
    plt.savefig(os.path.join(base_path, 'stereo_%s.jpg' % name), dpi=300)
    plt.close()


plt.figure(figsize=(8,8))
plt.imshow(getSample(hmi_dataset)[0], vmin=-1, vmax=1, cmap='gray')
plt.axis('off')
plt.tight_layout(0)
plt.savefig(os.path.join(base_path, 'hmi.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(8,8))
plt.imshow(getSample(hinode_dataset)[0], vmin=-1, vmax=1, cmap='gray')
plt.axis('off')
plt.tight_layout(0)
plt.savefig(os.path.join(base_path, 'hinode.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(8,8))
plt.imshow(getSample(q1_dataset)[0], vmin=-1, vmax=1, cmap='gray')
plt.axis('off')
plt.tight_layout(0)
plt.savefig(os.path.join(base_path, 'q1.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(8,8))
plt.imshow(getSample(q2_dataset)[0], vmin=-1, vmax=1, cmap='gray')
plt.axis('off')
plt.tight_layout(0)
plt.savefig(os.path.join(base_path, 'q2.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(8,8))
plt.imshow(getSample(film_dataset)[0], vmin=-1, vmax=1, cmap='gray')
plt.axis('off')
plt.tight_layout(0)
plt.savefig(os.path.join(base_path, 'film.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(8,8))
plt.imshow(getSample(ccd_dataset)[0], vmin=-1, vmax=1, cmap='gray')
plt.axis('off')
plt.tight_layout(0)
plt.savefig(os.path.join(base_path, 'ccd.jpg'), dpi=300)
plt.close()