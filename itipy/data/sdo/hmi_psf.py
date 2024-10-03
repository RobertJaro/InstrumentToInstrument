import numpy as np


def load_psf():
    coords = (np.stack(np.mgrid[:11, :11], 0) - 5) * 0.6
    r = np.sqrt(coords[0] ** 2 + coords[1] ** 2)
    phi = 2 * np.pi - np.arctan2(coords[0], coords[1])
    c_w = [0.641, 0.211, 0.066, 0.00467, 0.035]
    c_sig = [0.47, 1.155, 2.09, 4.42, 25.77]
    c_a = [0.131, 0.371, 0.54, 0.781, 0.115]
    c_u = [1, 1, 2, 1, 1]
    c_nu = np.rad2deg([-1.85, 2.62, -2.34, 1.255, 2.58])
    psf = np.sum([(1 + c_a[i] * np.cos(c_u[i] * phi + c_nu[i])) * c_w[i] * (
            1 / (2 * np.pi * c_sig[i] ** 2) * np.exp(-(r ** 2 / (2 * c_sig[i] ** 2)))) for i in range(5)], 0)
    return psf