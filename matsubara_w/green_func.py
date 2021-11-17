import numpy as np
from numpy.fft import fft, ifft
from scipy.linalg import lstsq

# Hilbert transform of Bethe lattice DOS
def bethe_gf(wn, sigma, mu, D):
    zeta = 1.j * wn + mu - sigma
    sq = np.sqrt((zeta)**2 - D**2)
    sig = np.sign(sq.imag * wn)
    return 2. / (zeta + sig * sq)

# Fourier transform
def ft(wn, g_tau, tau):
    exp = np.exp(1.j * np.outer(wn, tau))
    return np.dot(exp, g_tau)

# Inverse Fourier transform
def ift(wn, g_wn, tau, beta):
    exp = np.exp(-1.j * np.outer(tau, wn))
    return 1/beta * np.dot(exp, g_wn)