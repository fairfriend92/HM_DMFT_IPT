import numpy as np
from numpy.fft import fft, ifft, ifftshift

# Discrete Fourier transform
def ft(wn, g_tau, tau, beta):
    exp = np.exp(1.j * np.outer(wn, tau))   
    return np.dot(exp, g_tau)         
    
# Inverse discrete Fourier transform
def ift(wn, g_wn, tau, beta):
    # Subtract tail
    g_wn = g_wn - 1/(1.j*wn)   
    
    # Compute FT
    exp = np.exp(-1.j * np.outer(tau, wn))
    g_tau = np.dot(exp, g_wn) / beta
    
    # Add FT of tail   
    return g_tau - 0.5    