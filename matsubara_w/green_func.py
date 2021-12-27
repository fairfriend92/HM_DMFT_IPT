import numpy as np
import warnings
warnings.filterwarnings("error")

# Algorithm from Vidberg & Serene J. Low Temperature Phys. 29, 3-4, 179 (1977) 
def pade(u, z, zn, dz=1):    
    # Take positive frequencies
    zn = np.array([i for i in zn if i >= 0])    
    u = u[len(u)-len(zn):]      
    M = len(z)  
    N = len(zn)
    zn = np.zeros(N) + 1.j*zn
    z = z + 1.j*np.zeros(M)
    
    # Sample Matsubara frequency    
    idx = np.arange(0, N, dz)
    u = u[idx]
    zn = zn[idx]      
    N = len(zn)   
      
    # Compute coefficients
    g = np.zeros((N, N), dtype = complex)
    g[0] = u 
    for i in range(1, N):
        try:            
            g[i, i:] = (g[i-1, i-1] - g[i-1, i:]) / ((zn[i:] - zn[i-1])*g[i-1, i:])
        except RuntimeWarning as e: 
            print(e)
            break
    a = np.diag(g)
    
    # Recursion formula for continued fractions
    A = np.zeros((N+1, M), dtype = np.clongdouble)
    B = np.zeros((N+1, M), dtype = np.clongdouble)
    A[0] = np.zeros(M)
    A[1] = a[0] * np.ones(M)  
    B[0] = B[1] = np.ones(M)
    for i in range(2, N+1):
        A[i] = A[i-1] + (z - zn[i-2])*a[i-1]*A[i-2] 
        B[i] = B[i-1] + (z - zn[i-2])*a[i-1]*B[i-2]
    return A[N] / B[N]

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