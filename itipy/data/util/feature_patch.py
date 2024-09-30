from matplotlib.colors import Normalize
from skimage import exposure
from sunpy.map import Map, all_coordinates_from_map

from itipy.data.editor import NormalizeRadiusEditor, LimbDarkeningCorrectionEditor, KSOFilmPrepEditor, LoadFITSEditor, \
    ImageNormalizeEditor, NanEditor, ContrastNormalizeEditor, FeaturePatchEditor
import numpy as np

from matplotlib import pyplot as plt

s_map = Map('/Users/robert/PycharmProjects/InstrumentToInstrument/dataset/2011-01-04T00:00.fits')
s_map.peek()
s_map = FeaturePatchEditor(patch_shape=(256, 256)).call(s_map)
s_map.peek()
#plt.hist(np.ravel(s_map.data), bins=100)