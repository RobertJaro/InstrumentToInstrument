import matplotlib.pyplot as plt
import numpy as np
from skimage import restoration
from sunpy.map import Map

coords = (np.stack(np.mgrid[:11, :11], 0) - 5) * 0.6
r = np.sqrt(coords[0] ** 2 + coords[1] ** 2)
phi = 2 * np.pi - np.arctan2(coords[0], coords[1])
c_w = [0.641, 0.211, 0.066, 0.00467, 0.035]
c_sig = [0.47, 1.155, 2.09, 4.42, 25.77]
c_a = [0.131, 0.371, 0.54, 0.781, 0.115]
c_u = [1, 1, 2, 1, 1]
c_nu = np.rad2deg([-1.85, 2.62, -2.34, 1.255, 2.58])
psf = np.sum([(1 + c_a[i] * np.cos(c_u[i]*phi + c_nu[i])) * c_w[i] * (1 / (2 * np.pi * c_sig[i] ** 2) * np.exp(-(r**2/(2*c_sig[i]**2)))) for i in range(5)], 0)

hmi_map = Map('/Users/robert/PycharmProjects/InstrumentToInstrument/dataset/hmi.Ic_720s.20220101_190000_TAI.3.continuum.fits')
hmi_data = hmi_map.data[-1024 - 100 - 512:-1024 - 400, 400:600]
decon_data = Map('/Users/robert/PycharmProjects/InstrumentToInstrument/dataset/deconS.hmi.Ic_720s.20220101_190000_TAI.3.continuum.fits').data[-1024 - 100 - 512:-1024 - 400, 400:600]

cdecon_data = restoration.richardson_lucy(hmi_data, psf, iterations=30, clip=False)

vmin =0
vmax = np.nanmax(hmi_data)
extent = (0, hmi_data.shape[1] * hmi_map.scale[1].value, 0, hmi_data.shape[0] * hmi_map.scale[0].value)
fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True)
axs[0].imshow(hmi_data, cmap='gray', vmin=vmin, vmax=vmax, extent=extent, origin='lower')
axs[1].imshow(cdecon_data, cmap='gray', vmin=vmin, vmax=vmax, extent=extent, origin='lower')
axs[2].imshow(decon_data, cmap='gray', vmin=vmin, vmax=vmax, extent=extent, origin='lower')
axs[0].set_xlabel('X [arcsec]'), axs[1].set_xlabel('X [arcsec]'), axs[2].set_xlabel('X [arcsec]')
axs[0].set_ylabel('Y [arcsec]')
axs[0].set_title('original'), axs[1].set_title('custom deconv'), axs[2].set_title('JSOC deconv')
plt.show()