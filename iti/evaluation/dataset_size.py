import pandas

from iti.data.dataset import KSOFlatDataset, KSOFilmDataset, SDODataset, SOHODataset, STEREODataset, HMIContinuumDataset

train_months = list(range(2, 10))
test_months = [11, 12]

synoptic_train = KSOFlatDataset("/gpfs/gpfs0/robert.jarolim/data/iti/kso_q1", 1024, months=train_months)
synoptic_test = KSOFlatDataset("/gpfs/gpfs0/robert.jarolim/data/iti/kso_q1", 1024, months=test_months)
print('KSO Synoptic: %d / %d' % (len(synoptic_train), len(synoptic_test)))

q2_train = KSOFlatDataset('/gpfs/gpfs0/robert.jarolim/data/iti/kso_q2', 1024, months=train_months)
q2_test = KSOFlatDataset('/gpfs/gpfs0/robert.jarolim/data/iti/kso_q2', 1024, months=test_months)
print('KSO Q2: %d / %d' % (len(q2_train), len(q2_test)))

# film_train = KSOFilmDataset("/gss/r.jarolim/data/filtered_kso_plate", 1024, months=train_months)
# film_test = KSOFilmDataset("/gss/r.jarolim/data/filtered_kso_plate", 1024, months=test_months)
# print('Film: %d / %d' % (len(film_train), len(film_test)))

sdo_train = SDODataset("/gpfs/gpfs0/robert.jarolim/data/iti/sdo", patch_shape=(1024, 1024), resolution=2048,
                       months=train_months)
sdo_test = SDODataset("/gpfs/gpfs0/robert.jarolim/data/iti/sdo", patch_shape=(1024, 1024), resolution=2048,
                      months=test_months)
print('SDO: %d / %d' % (len(sdo_train), len(sdo_test)))

soho_train = SOHODataset("/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep", resolution=1024, months=train_months)
soho_test = SOHODataset("/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep", resolution=1024, months=test_months)
print('SOHO: %d / %d' % (len(soho_train), len(soho_test)))

stereo_train = STEREODataset("/gpfs/gpfs0/robert.jarolim/data/iti/stereo_iti2021_prep", patch_shape=(1024, 1024), months=train_months)
stereo_test = STEREODataset("/gpfs/gpfs0/robert.jarolim/data/iti/stereo_iti2021_prep", patch_shape=(1024, 1024), months=test_months)
print('STEREO: %d / %d' % (len(stereo_train), len(stereo_test)))

hmi_train = HMIContinuumDataset("/gpfs/gpfs0/robert.jarolim/data/iti/hmi_continuum", (512, 512), months=train_months)
hmi_test = HMIContinuumDataset("/gpfs/gpfs0/robert.jarolim/data/iti/hmi_continuum", (512, 512), months=test_months)
print('HMI: %d / %d' % (len(hmi_train), len(hmi_test)))

df = pandas.read_csv('/gpfs/gpfs0/robert.jarolim/data/iti/hinode_file_list.csv', index_col=False, parse_dates=['date'])
hinode_test = df[df.date.dt.month.isin(test_months)]
hinode_train = df[df.date.dt.month.isin(train_months)]
print('Hinode: %d / %d' % (len(hinode_train), len(hinode_test)))

