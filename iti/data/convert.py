from iti.data.dataset import HMIContinuumDataset, StorageDataset, HinodeDataset, SDODataset, SOHODataset, STEREODataset
from iti.data.editor import RandomPatchEditor

if __name__ == '__main__':
    # sdo_dataset = HMIContinuumDataset("/gss/r.jarolim/data/hmi_continuum/6173", (256, 256))
    # sdo_dataset = StorageDataset(sdo_dataset,
    #                              '/gss/r.jarolim/data/converted/hmi_train',
    #                              ext_editors=[])
    # sdo_dataset.convert(8)
    # hinode_dataset = StorageDataset(HinodeDataset("/gss/r.jarolim/data/hinode/level1"),
    #                                 '/gss/r.jarolim/data/converted/hinode_train',
    #                                 ext_editors=[])
    # hinode_dataset.convert(8)
    # sdo_dataset = SDODataset("/gss/r.jarolim/data/sdo/train", resolution=4096, patch_shape=(512, 512))
    # sdo_dataset = StorageDataset(sdo_dataset,
    #                              '/gss/r.jarolim/data/converted/sdo_fullres_train',
    #                              ext_editors=[])
    # sdo_dataset.convert(8)

    sdo_dataset = SDODataset("/gss/r.jarolim/data/sdo/valid", resolution=4096, patch_shape=(512, 512))
    sdo_dataset = StorageDataset(sdo_dataset,
                                 '/gss/r.jarolim/data/converted/sdo_fullres_valid',
                                 ext_editors=[])
    sdo_dataset.convert(8)

    # soho_dataset = SOHODataset("/gss/r.jarolim/data/soho/train", patch_shape=(256, 256))
    # storage_ds = StorageDataset(soho_dataset, '/gss/r.jarolim/data/converted/soho_train', ext_editors=[])
    # storage_ds.convert(8)
    # stereo_dataset = STEREODataset("/gss/r.jarolim/data/stereo_prep/train", patch_shape=(256, 256))
    # stereo_dataset = StorageDataset(stereo_dataset,
    #                              '/gss/r.jarolim/data/converted/stereo_train',
    #                                 ext_editors=[])
    # stereo_dataset.convert(12)
    #
    # stereo_dataset = STEREODataset("/gss/r.jarolim/data/stereo_prep/valid", patch_shape=(256, 256))
    # stereo_dataset = StorageDataset(stereo_dataset,
    #                                 '/gss/r.jarolim/data/converted/stereo_valid',
    #                                 ext_editors=[])
    # stereo_dataset.convert(12)