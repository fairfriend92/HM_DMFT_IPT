import numpy as np
from numpy.fft import fft, ifft

# Discrete Fourier transform
def ft(wn, g_tau, tau, beta):
    exp = np.exp(1.j * np.outer(wn, tau))   
    return np.dot(exp, g_tau) / np.sqrt(len(tau))      
    
# Inverse discrete Fourier transform
def ift(wn, g_wn, tau, beta, a=0.):
    # Subtract tail
    g_wn = g_wn - a/(1.j*wn)   
    
    # Compute FT
    exp = np.exp(-1.j * np.outer(tau, wn))
    g_tau = np.dot(exp, g_wn) / beta
    
    # Add FT of tail   
    return g_tau - a*0.5    

# Fast Fourier transform (not working!)    
def my_fft(wn, g_tau, tau, beta):
    tau = np.append([0], tau)                       # Add tau=0
    g_tau = np.append([0], g_tau)
    n = len(tau) 
    exp = np.exp(-1.j*np.pi*tau/beta)
    sum_pos = n * ifft(g_tau, n) * exp              # Positive Matsubara freq
    sum_neg = np.delete(fft(g_tau, n+1), 0) * exp   # Negative Matsubara freq, delete extra w(n=0) freq
    sum_neg = np.flip(sum_neg)                      # Sort in increasing order 
    
    return np.append(sum_neg, sum_pos)

# Inverse fast Fourier transform (must fix for arbitrary N) 
def my_ifft(wn, g_wn, tau, beta, a=0.):
    # Subtract tail
    g_wn = g_wn - a/(1.j*wn)                    
   
    # Compute FT
    n = int(len(wn)/2)  
    g_pos = g_wn[n:]                            # Positive frequencies G
    g_neg = np.append([0], np.flip(g_wn[:n]))   # Neg freq G, padded with n=0 freq and...
                                                # ...sorted in decreasing order
    sum_pos = np.delete(fft(g_pos, n), 0)       # Positive freq FFT, delete tau=0 element
    sum_neg = np.delete(n*ifft(g_neg, n), 0)    # Negative freq FFT, delete tau=0 element
    exp = np.exp(-1.j*np.pi*tau/beta)  
    g_tau = (sum_pos + sum_neg) * exp/beta  
    
    # Add FT of tail
    return g_tau - a*0.5