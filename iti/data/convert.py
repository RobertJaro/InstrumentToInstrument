from iti.data.dataset import HMIContinuumDataset, StorageDataset, HinodeDataset, SDODataset, SOHODataset, STEREODataset
from iti.data.editor import RandomPatchEditor

if __name__ == '__main__':
    ###########################################################
    # HMI
    ###########################################################
    # hmi_dataset = HMIContinuumDataset("/gss/r.jarolim/data/hmi_continuum/6173", (512, 512))
    # hmi_dataset = StorageDataset(hmi_dataset, '/gss/r.jarolim/data/converted/hmi_continuum', ext_editors=[])
    # hmi_dataset.convert(12)
    ###########################################################
    # Hinode
    ###########################################################
    # hinode_dataset = HinodeDataset("/gss/r.jarolim/data/hinode/level1")
    # hinode_dataset = StorageDataset(hinode_dataset, '/gss/r.jarolim/data/converted/hinode_continuum', ext_editors=[])
    # hinode_dataset.convert(12)
    ###########################################################
    # SDO
    ###########################################################
    sdo_dataset = SDODataset("/gss/r.jarolim/data/ch_detection", resolution=4096, patch_shape=(1024, 1024))
    sdo_dataset = StorageDataset(sdo_dataset, '/gss/r.jarolim/data/converted/sdo_4096', ext_editors=[])
    sdo_dataset.convert(12)
    ###########################################################
    sdo_dataset = SDODataset("/gss/r.jarolim/data/ch_detection", resolution=2048, patch_shape=(1024, 1024))
    sdo_dataset = StorageDataset(sdo_dataset, '/gss/r.jarolim/data/converted/sdo_2048', ext_editors=[])
    sdo_dataset.convert(12)
    ###########################################################
    # SOHO
    ###########################################################
    soho_dataset = SOHODataset("/gss/r.jarolim/data/soho_iti2021_prep")
    storage_ds = StorageDataset(soho_dataset, '/gss/r.jarolim/data/converted/soho_1024', ext_editors=[])
    storage_ds.convert(12)
    ###########################################################
    # STEREO
    ###########################################################
    stereo_dataset = STEREODataset("/gss/r.jarolim/data/stereo_iti2021_prep", patch_shape=(1024, 1024))
    stereo_dataset = StorageDataset(stereo_dataset, '/gss/r.jarolim/data/converted/stereo_1024', ext_editors=[])
    stereo_dataset.convert(12)