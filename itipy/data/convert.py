import pandas as pd

from itipy.data.dataset import HMIContinuumDataset, StorageDataset, HinodeDataset, SDODataset, SOHODataset, STEREODataset, \
    get_intersecting_files
from itipy.data.editor import RandomPatchEditor

if __name__ == '__main__':
    months = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
    base_path = '/gpfs/gpfs0/robert.jarolim/data'
    ###########################################################
    # HMI
    ###########################################################
    hmi_dataset = HMIContinuumDataset('%s/iti/hmi_continuum' % base_path, (512, 512))
    hmi_dataset = StorageDataset(hmi_dataset, '%s/converted/hmi_continuum' % base_path, ext_editors=[])
    hmi_dataset.convert(12)
    ###########################################################
    # Hinode
    ###########################################################
    df = pd.read_csv('/beegfs/home/robert.jarolim/data/ITI_converted/hinode_file_list.csv', index_col=False,
                     parse_dates=['date'])
    hinode_dataset = HinodeDataset(df.file)
    hinode_dataset = StorageDataset(hinode_dataset, '/beegfs/home/robert.jarolim/data/ITI_converted/hinode',
                                    ext_editors=[RandomPatchEditor((640, 640))])
    invalid_files = hinode_dataset.convert(12)
    df = df[~df.file.isin(invalid_files)]
    df.to_csv('/beegfs/home/robert.jarolim/data/ITI_converted/hinode_file_list.csv', index=False)
    ###########################################################
    # SDO
    ###########################################################
    sdo_files = get_intersecting_files("%s/iti/sdo" % base_path, ['171', '193', '211', '304', '6173'],
                                       ext='.fits', months=months)

    sdo_dataset = SDODataset(sdo_files, resolution=2048, patch_shape=(1024, 1024))
    sdo_dataset = StorageDataset(sdo_dataset, '%s/converted/sdo_2048' % base_path, ext_editors=[])
    sdo_dataset.convert(12)
    ###########################################################
    sdo_dataset = SDODataset(sdo_files, resolution=4096, patch_shape=(1024, 1024))
    sdo_dataset = StorageDataset(sdo_dataset, '%s/converted/sdo_4096' % base_path, ext_editors=[])
    sdo_dataset.convert(12)
    ###########################################################
    sdo_dataset = SDODataset(sdo_files, resolution=512)
    sdo_dataset = StorageDataset(sdo_dataset, '%s/converted/sdo_512' % base_path, ext_editors=[])
    sdo_dataset.convert(12)
    ###########################################################
    # SOHO
    ###########################################################
    soho_dataset = SOHODataset("%s/iti/soho_iti2021_prep" % base_path, months=months)
    storage_ds = StorageDataset(soho_dataset, '%s/converted/soho_1024' % base_path, ext_editors=[])
    storage_ds.convert(12)
    ###########################################################
    # STEREO
    ###########################################################
    stereo_dataset = STEREODataset("%s/iti/stereo_iti2021_prep" % base_path, months=months)
    stereo_dataset = StorageDataset(stereo_dataset, '%s/converted/stereo_1024_calibrated' % base_path,
                                    ext_editors=[])
    stereo_dataset.convert(12)
