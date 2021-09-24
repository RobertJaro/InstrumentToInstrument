import pandas

from iti.data.dataset import KSOFlatDataset, KSOFilmDataset, SDODataset, SOHODataset, STEREODataset, HMIContinuumDataset

synoptic_train = KSOFlatDataset("/gss/r.jarolim/data/kso_synoptic", 1024, months=list(range(11)))
synoptic_test = KSOFlatDataset("/gss/r.jarolim/data/kso_synoptic", 1024, months=[11, 12])
print('KSO Synoptic: %d / %d' % (len(synoptic_train), len(synoptic_test)))

q2_train = KSOFlatDataset('/gss/r.jarolim/data/kso_general/quality2', 1024, months=list(range(11)))
q2_test = KSOFlatDataset('/gss/r.jarolim/data/kso_general/quality2', 1024, months=[11, 12])
print('KSO Q2: %d / %d' % (len(q2_train), len(q2_test)))

film_train = KSOFilmDataset("/gss/r.jarolim/data/filtered_kso_plate", 1024, months=list(range(11)))
film_test = KSOFilmDataset("/gss/r.jarolim/data/filtered_kso_plate", 1024, months=[11, 12])
print('Film: %d / %d' % (len(film_train), len(film_test)))

sdo_train = SDODataset("/gss/r.jarolim/data/ch_detection", patch_shape=(1024, 1024), resolution=2048,
                           months=list(range(11)))
sdo_test = SDODataset("/gss/r.jarolim/data/ch_detection", patch_shape=(1024, 1024), resolution=2048,
                           months=[11, 12])
print('SDO: %d / %d' % (len(sdo_train), len(sdo_test)))

soho_train = SOHODataset("/gss/r.jarolim/data/soho_iti2021_prep", resolution=1024, months=list(range(11)))
soho_test = SOHODataset("/gss/r.jarolim/data/soho_iti2021_prep", resolution=1024, months=[11, 12])
print('SOHO: %d / %d' % (len(soho_train), len(soho_test)))

stereo_train = STEREODataset("/gss/r.jarolim/data/stereo_iti2021_prep", patch_shape=(1024, 1024), months=list(range(11)))
stereo_test = STEREODataset("/gss/r.jarolim/data/stereo_iti2021_prep", patch_shape=(1024, 1024), months=[11, 12])
print('STEREO: %d / %d' % (len(stereo_train), len(stereo_test)))

hmi_train = HMIContinuumDataset("/gss/r.jarolim/data/hmi_continuum/6173", (512, 512), months=list(range(11)))
hmi_test = HMIContinuumDataset("/gss/r.jarolim/data/hmi_continuum/6173", (512, 512), months=[11, 12])
print('HMI: %d / %d' % (len(hmi_train), len(hmi_test)))

df = pandas.read_csv('/gss/r.jarolim/data/hinode/file_list.csv', index_col=False, parse_dates=['date'])
hinode_test = df[(df.date.dt.month == 12) | (df.date.dt.month == 11)]
hinode_train = df[(df.date.dt.month != 12) & (df.date.dt.month != 11)]
print('Hinode: %d / %d' % (len(hinode_train), len(hinode_test)))

