
#%%
from datetime import datetime

from sunpy.net import Fido, attrs as a

#%%
result = Fido.search(a.Time(datetime(2014, 1, 1), datetime(2014, 1, 2)), a.Instrument('SOT'))
