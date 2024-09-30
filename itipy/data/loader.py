import numpy as np
from astropy.visualization import AsinhStretch
from torch.utils.data import Dataset

from itipy.data.editor import LoadMapEditor, NormalizeRadiusEditor, AIAPrepEditor, EITCheckEditor, SOHOFixHeaderEditor, \
    RemoveOffLimbEditor, \
    KSOPrepEditor, LimbDarkeningCorrectionEditor, MapToDataEditor, ImageNormalizeEditor, NanEditor, ReshapeEditor, \
    MapImageNormalizeEditor


class BaseLoader(Dataset):

    def __init__(self, editors):
        self.editors = editors

    def __call__(self, data):
        kwargs = {}
        for editor in self.editors:
            data, kwargs = editor.convert(data, **kwargs)
        return data


class AIAMapLoader(BaseLoader):

    def __init__(self, resolution=2048, calibration='auto'):
        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution),
                   AIAPrepEditor(calibration=calibration)]
        super().__init__(editors)


class HMIMapLoader(BaseLoader):

    def __init__(self, resolution=2048):
        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution),
                   RemoveOffLimbEditor()]
        super().__init__(editors)


class EITMapLoader(BaseLoader):

    def __init__(self, resolution=1024):
        editors = [LoadMapEditor(),
                   EITCheckEditor(),
                   SOHOFixHeaderEditor(),
                   NormalizeRadiusEditor(resolution)]
        super().__init__(editors)


class MDIMapLoader(BaseLoader):

    def __init__(self, resolution=1024):
        editors = [LoadMapEditor(),
                   SOHOFixHeaderEditor(),
                   NormalizeRadiusEditor(resolution),
                   RemoveOffLimbEditor()]
        super().__init__(editors)


class KSOFlatLoader(BaseLoader):

    def __init__(self, resolution, **kwargs):
        editors = [LoadMapEditor(),
                   KSOPrepEditor(),
                   NormalizeRadiusEditor(resolution, 0),
                   LimbDarkeningCorrectionEditor(),
                   MapImageNormalizeEditor(0.65, 1.5, stretch=AsinhStretch(0.5))
                   ]
        super().__init__(editors, **kwargs)


class HMIContinuumLoader(BaseLoader):

    def __init__(self, resolution=1024):
        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution),
                   RemoveOffLimbEditor(fill_value=np.nan),
                   ]
        super().__init__(editors)
