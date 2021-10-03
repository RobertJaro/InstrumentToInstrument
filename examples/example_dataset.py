from astropy.visualization import ImageNormalize, LinearStretch

from iti.data.dataset import BaseDataset
from iti.data.editor import LoadMapEditor, NormalizeRadiusEditor, RemoveOffLimbEditor, MapToDataEditor, NanEditor, \
    NormalizeEditor, ReshapeEditor


class HMIDataset(BaseDataset):

    def __init__(self, path, resolution=2048, ext='.fits', **kwargs):
        norm = ImageNormalize(vmin=-1000, vmax=1000, stretch=LinearStretch(), clip=True)
        editors = [
            # open FITS
            LoadMapEditor(),
            # normalize rad
            NormalizeRadiusEditor(resolution),
            # truncate off limb (optional)
            RemoveOffLimbEditor(),
            # get data from SunPy map
            MapToDataEditor(),
            # replace NaN with 0
            NanEditor(),
            # normalize data to [-1, 1]
            NormalizeEditor(norm),
            # change data to channel first format
            ReshapeEditor((1, resolution, resolution))]
        super().__init__(path, editors=editors, ext=ext, **kwargs)
