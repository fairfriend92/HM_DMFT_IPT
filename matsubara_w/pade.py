import numpy as np
import warnings
warnings.filterwarnings("error")

# Algorithm from Vidberg & Serene J. Low Temperature Phys. 29, 3-4, 179 (1977) 
def my_pade(u, z, zn, dz=4):
    M = len(z)    
    zn = 1.j*zn
    z = z + 1.j*np.zeros(M)

    # Sample Matsubara frequency
    idx = np.arange(int(dz/2), len(zn), dz)
    u = u[idx]
    zn = zn[idx]  
    N = len(zn)
    #print(idx)
    #print(zn)
    
    g = np.zeros((N, N), dtype = complex)
    g[0] = u 
    for i in range(1, N):
        try:            
            g[i, i:] = (g[i-1, i-1] - g[i-1, i:]) / ((zn[i:] - zn[i-1])*g[i-1, i:])
        except RuntimeWarning as e: 
            print(e)
            break
    a = np.diag(g)
    A = np.zeros((N, M), dtype = complex)
    B = np.zeros((N, M), dtype = complex)
    A[0] = np.zeros(M)
    A[1] = a[0] * np.ones(M)
    B[0] = B[1] = np.ones(M)
    for i in range(2, N):
        A[i] = A[i-1] + (z - zn[i-1])*a[i]*A[i-2]
        B[i] = B[i-1] + (z - zn[i-1])*a[i]*B[i-2]
    return A[-1] / B[-1]
    
'''
def pade_coefficients(gf, wn):    
    g = np.zeros((len(gf), len(gf)), dtype=np.complex)
    g[0] = gf
    for i in range(1, len(gf)):
        g[i, i:] = 1j * (g[i - 1, i - 1] - g[i - 1, i:]) / 
                   ((wn[i - 1] - wn[i:])*g[i - 1, i:])

    return np.diag(g)

def pade_rec(pc, # Pade coefficient
             w, wn):    
    an_1 = 0.
    an = pc[0]
    bn = 1.
    bn_1 = 1.
    iwn = 1j * wn
    for i in range(len(pc) - 2):
        anp = an + (w - iwn[i]) * pc[i + 1] * an_1
        bnp = bn + (w - iwn[i]) * pc[i + 1] * bn_1
        an_1, an = an, anp
        bn_1, bn = bn, bnp
    return an / bn
    

# Continuate the Green Function by Pade
def pade_continuation(gf, wn, w, 
                      w_set = None):  # Index of points to sample for continuation
    if w_set is None:
        w_set = np.arange(len(wn))
    elif isinstance(w_set, int):
        w_set = np.arange(w_set)

    n = int(len(wn)/2)
    pc = pade_coefficients(gf[n:], wn[n:])
    g_real = pade_rec(pc, w, wn)

    return g_real
'''