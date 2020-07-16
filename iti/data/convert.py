from iti.data.dataset import HMIContinuumDataset, StorageDataset, HinodeDataset
from iti.data.editor import RandomPatchEditor

if __name__ == '__main__':
    sdo_dataset = HMIContinuumDataset("/gss/r.jarolim/data/hmi_continuum/6173", (256, 256))
    sdo_dataset = StorageDataset(sdo_dataset,
                                 '/gss/r.jarolim/data/converted/hmi_train',
                                 ext_editors=[])
    sdo_dataset.convert(8)
    # hinode_dataset = StorageDataset(HinodeDataset("/gss/r.jarolim/data/hinode/level1"),
    #                                 '/gss/r.jarolim/data/converted/hinode_train',
    #                                 ext_editors=[])
    # hinode_dataset.convert(8)