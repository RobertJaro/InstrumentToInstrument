from iti.data.dataset import HMIContinuumDataset, StorageDataset, HinodeDataset, SDODataset, SOHODataset, STEREODataset
from iti.data.editor import RandomPatchEditor

if __name__ == '__main__':
    # hmi_dataset = HMIContinuumDataset("/gss/r.jarolim/data/hmi_continuum/6173", (256, 256))
    # hmi_dataset = StorageDataset(hmi_dataset,
    #                              '/gss/r.jarolim/data/converted/hmi_train',
    #                              ext_editors=[])
    # hmi_dataset.convert(12)
    #
    # hinode_dataset = StorageDataset(HinodeDataset("/gss/r.jarolim/data/hinode/level1"),
    #                                 '/gss/r.jarolim/data/converted/hinode_train',
    #                                 ext_editors=[])
    # hinode_dataset.convert(8)
    ###########################################################
    # SDO
    ###########################################################
    sdo_dataset = SDODataset("/gss/r.jarolim/data/ch_detection", resolution=4096, patch_shape=(1024, 1024), months=list(range(1,11)))
    sdo_dataset = StorageDataset(sdo_dataset,
                                 '/gss/r.jarolim/data/converted/sdo_4096_train',
                                 ext_editors=[])
    sdo_dataset.convert(12)
    ###########################################################
    sdo_dataset = SDODataset("/gss/r.jarolim/data/ch_detection", resolution=4096, patch_shape=(1024, 1024),
                             months=list(range(11, 13)))
    sdo_dataset = StorageDataset(sdo_dataset,
                                 '/gss/r.jarolim/data/converted/sdo_4096_valid',
                                 ext_editors=[])
    sdo_dataset.convert(12)
    ###########################################################
    sdo_dataset = SDODataset("/gss/r.jarolim/data/ch_detection", resolution=2048, patch_shape=(1024, 1024),
                             months=list(range(1, 11)))
    sdo_dataset = StorageDataset(sdo_dataset,
                                 '/gss/r.jarolim/data/converted/sdo_2048_train',
                                 ext_editors=[])
    sdo_dataset.convert(12)
    ###########################################################
    sdo_dataset = SDODataset("/gss/r.jarolim/data/ch_detection", resolution=2048, patch_shape=(1024, 1024),
                             months=list(range(11, 13)))
    sdo_dataset = StorageDataset(sdo_dataset,
                                 '/gss/r.jarolim/data/converted/sdo_2048_valid',
                                 ext_editors=[])
    sdo_dataset.convert(12)
    ###########################################################
    # SOHO
    ###########################################################
    # soho_dataset = SOHODataset("/gss/r.jarolim/data/soho/train")
    # storage_ds = StorageDataset(soho_dataset, '/gss/r.jarolim/data/converted/soho_train', ext_editors=[])
    # storage_ds.convert(12)
    ###########################################################
    # soho_dataset = SOHODataset("/gss/r.jarolim/data/soho/valid")
    # storage_ds = StorageDataset(soho_dataset, '/gss/r.jarolim/data/converted/soho_valid', ext_editors=[])
    # storage_ds.convert(12)
    ###########################################################
    # STEREO
    ###########################################################
    # stereo_dataset = STEREODataset("/gss/r.jarolim/data/stereo_prep/train", patch_shape=(256, 256))
    # stereo_dataset = StorageDataset(stereo_dataset,
    #                              '/gss/r.jarolim/data/converted/stereo_train',
    #                                 ext_editors=[])
    # stereo_dataset.convert(12)
    ###########################################################
    # stereo_dataset = STEREODataset("/gss/r.jarolim/data/stereo_prep/valid", patch_shape=(256, 256))
    # stereo_dataset = StorageDataset(stereo_dataset,
    #                                 '/gss/r.jarolim/data/converted/stereo_valid',
    #                                 ext_editors=[])
    # stereo_dataset.convert(12)