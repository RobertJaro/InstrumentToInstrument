import glob
import os

import matplotlib.pyplot as plt
import numpy as np

path = '/Users/robert/PycharmProjects/InstrumentToInstrument/result/stereo_mag_v11/series'

iti_imgs = sorted(glob.glob(os.path.join(path, '*_iti.jpg')))
soho_imgs = sorted(glob.glob(os.path.join(path, '*_soho.jpg')))
euv_imgs = sorted(glob.glob(os.path.join(path, '*_195.jpg')))

for i, (iti_img, soho_img, euv_img) in enumerate(zip(iti_imgs, soho_imgs, euv_imgs)):
    fig, axs = plt.subplots(1, 3, figsize=(11, 4))
    [ax.set_axis_off() for ax in np.ravel(axs)]
    axs[0].imshow(plt.imread(euv_img), origin='lower')
    axs[0].set_title('STEREO - EUV')
    axs[1].imshow(plt.imread(iti_img), origin='lower')
    axs[1].set_title('ITI')
    axs[2].imshow(plt.imread(soho_img), origin='lower')
    axs[2].set_title('SOHO')
    plt.tight_layout()
    fig.savefig('/Users/robert/PycharmProjects/InstrumentToInstrument/result/stereo_mag_v11/series/frame_%d.jpg' % i, dpi=300)
    plt.close(fig)