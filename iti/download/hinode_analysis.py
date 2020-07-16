# %%
from datetime import datetime, timedelta

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import getdata
from sunpy.coordinates import sun
from sunpy.map import Map
from sunpy.net import Fido, attrs as a
from matplotlib import pyplot as plt
from reproject import reproject_interp

#%%
hinode_map = Map('/Users/robert/PycharmProjects/InstrumentToInstrument/dataset/FG20120204_122341.2.fits')#/Users/robert/PycharmProjects/InstrumentToInstrument/dataset/FG20120127_222709.9.fits')#'/Users/robert/PycharmProjects/InstrumentToInstrument/dataset/FG20121115_143051.0.fits')
hinode_map.peek()

#%%
result = Fido.search(a.Time(hinode_map.date, hinode_map.date + timedelta(minutes=10)), a.Instrument('HMI'), a.vso.Physobs('intensity'))
downloaded_files = Fido.fetch(result[:, 0])

#%%
hmi_map = Map(downloaded_files[0])
hmi_map.peek()

#%%
output, footprint = reproject_interp(hinode_map, hmi_map.wcs, hmi_map.data.shape)
hinode_out = Map(output, hmi_map.wcs)
hinode_out.plot_settings['cmap'] = hinode_map.plot_settings['cmap']

#%%
plt.subplot(121)
hinode_map.plot(cmap='gray')
plt.subplot(122)
hmi_map.submap(hinode_map.bottom_left_coord, hinode_map.top_right_coord).plot()
hinode_map.plot(alpha=0)

#%%
comp_map = Map(hmi_map, hinode_map, composite=True)
comp_map.set_alpha(1, 1)
comp_map.peek()

#%%
plt.subplot(111, projection=hmi_map)
hmi_map.plot()
hinode_map.plot()


#%%
srpp = hmi_map.scale[0] / hmi_map.rsun_obs * (hmi_map.data.shape[0] * u.pix)

#%%
s_map = hinode_map

scale_factor = s_map.scale[0] / 0.15
s_map = s_map.rotate(recenter=True, scale=scale_factor.value, missing=s_map.min(), order=3)
s_map.meta['r_sun'] = s_map.rsun_obs.value / s_map.meta['cdelt1']

# arcs_frame = s_map.rsun_obs * 1
# s_map = s_map.submap(SkyCoord([-arcs_frame, arcs_frame] * u.arcsec,
#                               [-arcs_frame, arcs_frame] * u.arcsec,
#                               frame=s_map.coordinate_frame))