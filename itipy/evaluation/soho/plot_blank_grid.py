import numpy as np
from matplotlib import pyplot as plt
from sunpy.visualization.colormaps import cm

from itipy.data.editor import sdo_norms, stereo_norms

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(np.ones((2000, 2000)) * np.nan, origin='lower', extend=[-1000, 1000, -1000, 1000], cmap='gray')

ax.set_xticks([-1000, -500, 0, 500, 1000])
ax.set_yticks([-1000, -500, 0, 500, 1000])

ax.set_xlabel('Helioprojective Longitude [arcsec]')
ax.set_ylabel('Helioprojective Latitude [arcsec]')

plt.tight_layout()
plt.savefig('/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3/blank_grid_1000.png', dpi=300, transparent=True)
plt.close()


fig, ax = plt.subplots(figsize=(3, 3))

im = ax.imshow(np.ones((1100, 1100)) * np.nan, origin='lower', extend=[-1100, 1100, -1100, 1100], cmap='gray')

ax.set_xticks([-1000, -500, 0, 500, 1000])
ax.set_yticks([-1000, -500, 0, 500, 1000])

ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_xlim(-1100, 1100)
ax.set_ylim(-1100, 1100)

plt.tight_layout()
plt.savefig('/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3/blank_grid_1100.png', dpi=300, transparent=True)
plt.close()

fig, ax = plt.subplots(figsize=(4, 3))

im = ax.imshow(np.ones((2000, 2000)) * np.nan, origin='lower', extend=[-111, 111, -111, 111], cmap='gray')

ax.set_xticks([-100, -50, 0, 50, 100])
ax.set_yticks([-100, -50, 0, 50, 100])
# ax.set_yticklabels([])
ax.set_xlim(-111, 111)
ax.set_ylim(-111, 111)

ax.set_xlabel('Helioprojective Longitude [arcsec]')
ax.set_ylabel('Helioprojective Latitude [arcsec]')

plt.tight_layout()
plt.savefig('/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3/blank_grid_222.png', dpi=300, transparent=True)
plt.close()

fig, ax = plt.subplots(figsize=(3, 3))

im = ax.imshow(np.ones((800, 800)) * np.nan, origin='lower', extend=[0, 800, 0, 800], cmap='gray')

ax.set_xticks([0, 200, 400, 600, 800])
ax.set_yticks([0, 200, 400, 600, 800])

ax.set_yticklabels([])
ax.set_xticklabels([])

# ax.set_xlabel('Helioprojective Longitude [arcsec]')
# ax.set_ylabel('Helioprojective Latitude [arcsec]')

plt.tight_layout()
plt.savefig('/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3/blank_grid_800.png', dpi=300, transparent=True)
plt.close()

fig, ax = plt.subplots(figsize=(3, 3))

im = ax.imshow(np.ones((13, 13)) * np.nan, origin='lower', extend=[0, 13, 0, 13], cmap='gray')

ax.set_xticks([0, 6, 12])
ax.set_yticks([0, 6, 12])
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xlim(0, 13)
ax.set_ylim(0, 13)

plt.tight_layout()
plt.savefig('/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3/blank_grid_13.png', dpi=300, transparent=True)
plt.close()

# plot colorbars

# 304
im = plt.imshow(np.ones((2000, 2000)) * np.nan, origin='lower', cmap=cm.sdoaia304, norm=sdo_norms[304])
plt.close()
fig, ax = plt.subplots(figsize=(1, 2))
cbar = plt.colorbar(im, cax=ax, orientation='vertical', label='Intensity [DN/s]', ticks=[0, 100, 1000, 8000])
cbar.set_ticklabels(['0', '1e2', '1e3', '8e3'])
plt.tight_layout()
plt.savefig('/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3/colorbar_304.png', dpi=300, transparent=True)
plt.close()

# 171
im = plt.imshow(np.ones((2000, 2000)) * np.nan, origin='lower', cmap=cm.sdoaia171, norm=sdo_norms[171])
plt.close()
fig, ax = plt.subplots(figsize=(1, 2))
cbar = plt.colorbar(im, cax=ax, orientation='vertical', label='Intensity [DN/s]', ticks=[0, 100, 1000, 8000])
cbar.set_ticklabels(['0', '1e2', '1e3', '8e3'])
plt.tight_layout()
plt.savefig('/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3/colorbar_171.png', dpi=300, transparent=True)
plt.close()

# 211
im = plt.imshow(np.ones((2000, 2000)) * np.nan, origin='lower', cmap=cm.sdoaia211, norm=sdo_norms[211])
plt.close()
fig, ax = plt.subplots(figsize=(1, 2))
cbar = plt.colorbar(im, cax=ax, orientation='vertical', label='Intensity [DN/s]', ticks=[0, 100, 1000, 5000])
cbar.set_ticklabels(['0', '1e2', '1e3', '5e3'])
plt.tight_layout()
plt.savefig('/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3/colorbar_211.png', dpi=300, transparent=True)
plt.close()

# 193
im = plt.imshow(np.ones((2000, 2000)) * np.nan, origin='lower', cmap=cm.sdoaia193, norm=sdo_norms[193])
plt.close()
fig, ax = plt.subplots(figsize=(1, 2))
cbar = plt.colorbar(im, cax=ax, orientation='vertical', label='Intensity [DN/s]', ticks=[0, 100, 1000, 9000])
cbar.set_ticklabels(['0', '1e2', '1e3', '9e3'])
plt.tight_layout()
fig.savefig('/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3/colorbar_193.png', dpi=300, transparent=True)
plt.close()

# mag
im = plt.imshow(np.ones((2000, 2000)) * np.nan, origin='lower', cmap='gray', vmin=-3000, vmax=3000)
plt.close()
fig, ax = plt.subplots(figsize=(1.18, 2))
cbar = plt.colorbar(im, cax=ax, orientation='vertical', label='B LOS [G]', ticks=[-3000, -1500, 0, 1500, 3000])
cbar.set_ticklabels(['-3e3', '-1.5e3', '0', '1.5e3', '3e3'])
plt.tight_layout()
fig.savefig('/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3/colorbar_mag.png', dpi=300, transparent=True)
plt.close()


# horizontal colorbars
# mag

im = plt.imshow(np.ones((2000, 2000)) * np.nan, origin='lower', cmap='gray', vmin=-2000, vmax=2000)
plt.close()

fig, ax = plt.subplots(figsize=(3, .9))
cbar = plt.colorbar(im, cax=ax, orientation='horizontal',
                    label='B LOS [G]', ticks=[-2000, -1000, 0, 1000, 2000],)
cbar.set_ticklabels(['', '-1e3', '0', '1e3', ''])
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
# increase font size
cbar.ax.tick_params(labelsize=12)
cbar.ax.xaxis.label.set_size(14)

plt.tight_layout()
fig.savefig('/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3/colorbar_mag_horizontal.png', dpi=300, transparent=True)
plt.close()


# EUVI 304

im = plt.imshow(np.ones((2000, 2000)) * np.nan, origin='lower', norm=stereo_norms[304], cmap=cm.euvi304)
plt.close()

fig, ax = plt.subplots(figsize=(3, .9))
cbar = plt.colorbar(im, cax=ax, orientation='horizontal',
                    label=r'EUVI 304 Å [DN/s]', ticks=[0, 100, 1000, 10000])
cbar.set_ticklabels(['0', '1e2', '1e3', '1e4'])
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
# increase font size
cbar.ax.tick_params(labelsize=12)
cbar.ax.xaxis.label.set_size(14)

plt.tight_layout()
fig.savefig('/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3/colorbar_304_horizontal.png', dpi=300, transparent=True)
plt.close()

# dummy gray from 0 to 1
im = plt.imshow(np.ones((2000, 2000)) * np.nan, origin='lower', cmap='gray', vmin=0, vmax=1)
plt.close()

fig, ax = plt.subplots(figsize=(3, .9))
cbar = plt.colorbar(im, cax=ax, orientation='horizontal',
                    label='Normalized Intensity', ticks=[0, 0.5, 1])

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')

plt.tight_layout()
fig.savefig('/beegfs/home/robert.jarolim/iti_evaluation/soho_sdo_v3/colorbar_dummy_horizontal.png', dpi=300, transparent=True)
plt.close()
