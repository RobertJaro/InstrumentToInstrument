from iti.data.dataset import HMIContinuumDataset, StorageDataset, HinodeDataset, SDODataset, SOHODataset, STEREODataset, \
    get_intersecting_files

if __name__ == '__main__':
    months = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
    ###########################################################
    # HMI
    ###########################################################
    hmi_dataset = HMIContinuumDataset('/gpfs/gpfs0/robert.jarolim/data/iti/hmi_continuum', (512, 512))
    hmi_dataset = StorageDataset(hmi_dataset, '/gpfs/gpfs0/robert.jarolim/data/converted/hmi_continuum', ext_editors=[])
    hmi_dataset.convert(12)
    ###########################################################
    # Hinode
    ###########################################################
    hinode_dataset = HinodeDataset("/gpfs/gpfs0/robert.jarolim/data/iti/hinode_iti2022_prep")
    hinode_dataset = StorageDataset(hinode_dataset, '/gpfs/gpfs0/robert.jarolim/data/converted/hinode_continuum',
                                    ext_editors=[])
    hinode_dataset.convert(12)
    ###########################################################
    # SDO
    ###########################################################
    sdo_files = get_intersecting_files("/gpfs/gpfs0/robert.jarolim/data/iti/sdo", ['171', '193', '211', '304', '6173'],
                                       ext='.fits', months=months)

    sdo_dataset = SDODataset(sdo_files, resolution=2048, patch_shape=(1024, 1024))
    sdo_dataset = StorageDataset(sdo_dataset, '/gpfs/gpfs0/robert.jarolim/data/converted/sdo_2048', ext_editors=[])
    sdo_dataset.convert(12)
    ###########################################################
    sdo_dataset = SDODataset(sdo_files, resolution=4096, patch_shape=(1024, 1024))
    sdo_dataset = StorageDataset(sdo_dataset, '/gpfs/gpfs0/robert.jarolim/data/converted/sdo_4096', ext_editors=[])
    sdo_dataset.convert(12)
    ###########################################################
    sdo_dataset = SDODataset(sdo_files, resolution=512)
    sdo_dataset = StorageDataset(sdo_dataset, '/gpfs/gpfs0/robert.jarolim/data/converted/sdo_512', ext_editors=[])
    sdo_dataset.convert(12)
    ###########################################################
    # SOHO
    ###########################################################
    soho_dataset = SOHODataset("/gpfs/gpfs0/robert.jarolim/data/iti/soho_iti2021_prep", months=months)
    storage_ds = StorageDataset(soho_dataset, '/gpfs/gpfs0/robert.jarolim/data/converted/soho_1024', ext_editors=[])
    storage_ds.convert(12)
    ###########################################################
    # STEREO
    ###########################################################
    stereo_dataset = STEREODataset("/gpfs/gpfs0/robert.jarolim/data/iti/stereo_iti2021_prep", months=months)
    stereo_dataset = StorageDataset(stereo_dataset, '/gpfs/gpfs0/robert.jarolim/data/converted/stereo_1024_calibrated',
                                    ext_editors=[])
    stereo_dataset.convert(12)
