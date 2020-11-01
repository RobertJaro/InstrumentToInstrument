from iti.data.editor import LoadMapEditor, KSOPrepEditor, NormalizeRadiusEditor, MapToDataEditor, ImageNormalizeEditor, \
    ContrastNormalizeEditor, NormalizeExposureEditor
from matplotlib import pyplot as plt
import numpy as np
from astropy import units as u

f = '/gss/r.jarolim/data/kso_general/quality2/kanz_halph_fi_20180907_093533.fts.gz'



s_map, _ = LoadMapEditor().call(f)
s_map = KSOPrepEditor().call(s_map)
s_map = NormalizeRadiusEditor(512).call(s_map)
s_map = NormalizeExposureEditor(2 * u.ms).call(s_map)
data, _ = MapToDataEditor().call(s_map)

plt.hist(np.ravel(data), 100)
plt.savefig('/gss/r.jarolim/data/kso_general/q2_hist.jpg')
plt.close()

normalized_data = ImageNormalizeEditor(150, 1000).call(data)

plt.imshow(normalized_data, cmap='gray', vmin=-1, vmax=1)
plt.savefig('/gss/r.jarolim/data/kso_general/q2_smaple.jpg')
plt.close()
