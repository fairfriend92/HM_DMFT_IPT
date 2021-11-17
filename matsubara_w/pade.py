import numpy as np

# Pade Analytical Continuation
# Algorithm from Vidberg & Serene J. Low Temperature Phys. 29, 3-4, 179 (1977)
def pade_coefficients(gf, wn):    
    g = np.zeros((len(gf), len(gf)), dtype=np.complex)
    g[0] = gf
    for i in range(1, len(gf)):
        g[i, i:] = 1j * (g[i - 1, i - 1] / g[i - 1, i:] -
                         1.) / (wn[i - 1] - wn[i:])

    return np.diag(g)

# Pade recursion formula for continued fractions
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

    pc = pade_coefficients(gf[w_set], wn[w_set])
    g_real = pade_rec(pc, w, wn[w_set])

    return g_real