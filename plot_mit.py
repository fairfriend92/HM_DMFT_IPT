# -*- coding: utf-8 -*-

'''
===================================
The Metal Mott Insulator transition
===================================

Using a real frequency IPT solver, follow the spectral function 
along the metal to insulator transition.
'''

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pylab as plt
import scipy.signal as signal

'''
Fermi Dirac distribution
'''
def fermi_dist(energy, beta):
    exponent = np.asarray(beta * energy).clip(-600, 600)
    return 1. / (np.exp(exponent) + 1)


'''
Imaginary part of self-energy according to IPT 
'''
def ph_hf_sigma(Aw, nf, U):
    # Because of particle-hole symmetry at half-fill 
    # in the single band one can work with A^+ only

    Ap = Aw * nf
    # Convolution A^+ * A^+ using FFT 
    App = signal.fftconvolve(Ap, Ap, mode='same')
    # Convolution A^-(-w) * App
    Appp = signal.fftconvolve(Ap, App, mode='same')
    return -np.pi * U**2 * (Appp + Appp[::-1])

'''
DMFT Loop for the single band Hubbard Model at Half-Filling
'''
def ss_dmft_loop(gloc, w, u_int, beta, conv):
    # gloc  Local Green's function to use as seed   (complex 1D ndarray) 
    # sigma Self-energy                             (complex 1D ndarray)
    # w     Real frequency points                   (real 1D ndarray) 
    # u_int On site interaction                     (float)
    # beta  1/T                                     (float)
    # conv  Convergence criteria                    (float)
  
    dw = w[1] - w[0]
    eta = 2j * dw
    nf = fermi_dist(w, beta)

    converged = False
    while not converged:
        gloc_old = gloc.copy()
        
        # Self-consistency
        g0 = 1 / (w + eta - .25 * gloc)
        
        # Spectral-function of Weiss field
        A0 = -g0.imag / np.pi   

        # Imaginary part of 2nd order diagram 
        isi = ph_hf_sigma(A0, nf, u_int) * dw * dw
        isi = 0.5 * (isi + isi[::-1])

        # Kramers-Kronig relation 
        hsi = -signal.hilbert(isi, len(isi) * 4)[:len(isi)].imag
        
        # Self-energy
        sigma = hsi + 1j * isi

        # Semi-circle Hilbert Transform
        gloc = semi_circle_hiltrans(w - sigma)
        converged = np.allclose(gloc, gloc_old, atol=conv)

    return gloc, sigma

'''
def semi_circle(energy, hopping):
    """Bethe lattice in inf dim density of states"""
    energy = np.asarray(energy).clip(-2 * hopping, 2 * hopping)
    return np.sqrt(4 * hopping**2 - energy**2) / (2 * np.pi * hopping**2)
'''

'''
Analytic continuation of Matsubara Green fucntion. Applies
Hilbert transform to the DOS of the Bethe lattice.
'''
def semi_circle_hiltrans(zeta, D=1):
    # D     Bandwidth
    # zeta  Frequencies in the Z plane
    
    sqr = np.sqrt(zeta**2 - D**2)
    sqr = np.sign(sqr.imag) * sqr
    return 2 * (zeta - sqr) / D**2

'''
MAIN LOOP
'''
#w = np.linspace(-25, 25, 2**14)
#urange = [0.2, 0.4, 0.65, 0.8, 1.0, 1.2, 1.5,1.7,1.9,2.0,2.2,2.4,2.5,2.75,3.0,3.25,3.5,3.7,3.9,4.0,4.5,5.0]

w = np.arange(-20,20,0.01)              # Frequencies  
gloc = semi_circle_hiltrans(w + 1e-3j)  # Local Green function
beta = 100.0                            # 1/temp
urange = [0.5,1.0,1.5]                  # On site interaction term   
            
#plt.close('all')

for i, U in enumerate(urange):
    print("U=%s"%(U))
    gloc, sigma_loc = ss_dmft_loop(gloc, w, U, beta, 1e-10)
    plt.gca().set_prop_cycle(None)
    shift = -2.1 * i
    plt.plot(w, shift + -gloc.imag)
    plt.axhline(shift, color='k', lw=0.5)
    np.savetxt("Green_tp_%s_U%s.txt"%(0.0,U),np.transpose([w,gloc.real,gloc.imag]))
    np.savetxt("Sigma_tp_%s_U%s.txt"%(0.0,U),np.transpose([w,sigma_loc.real,sigma_loc.imag]))
    
plt.xlabel(r'$\omega$')
plt.xlim([-4, 4])
plt.ylim([shift, 2.1])
plt.yticks(0.5 - 2.1 * np.arange(len(urange)), ['U=' + str(u) for u in urange])
plt.savefig("DOS.png")
#plt.show()
