import numpy as np

# Discrete Fourier transform
def ft(wn, g_tau, tau, beta, a=1.):
    exp = np.exp(1.j * np.outer(wn, tau))   
    return np.dot(exp, g_tau) * beta / len(tau) # Product of normalization factor of DFT and IFT...
                                                # ...must be 1/N    
# Inverse discrete Fourier transform
def ift(wn, g_wn, tau, beta, a=1.):
    # Subtract tail
    g_wn = g_wn - a/(1.j*wn)   
    
    # Compute FT
    exp = np.exp(-1.j * np.outer(tau, wn))
    g_tau = np.dot(exp, g_wn) / beta
        
    # Add FT of tail   
    return g_tau - a*0.5    