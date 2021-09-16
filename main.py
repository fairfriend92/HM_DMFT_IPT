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

w = np.arange(-20, 20, 0.01)            # Frequencies  
gloc = semi_circle_hiltrans(w + 1e-3j)  # Local Green function
beta = 100.0                            # 1/temp
urange = [0.5, 1.0, 1.5, 4.0]           # On site interaction term   

shift_g = 2
shift_s = 45
  
for i, U in enumerate(urange):
    print("U=%s"%(U))
    gloc, sigma_loc = ss_dmft_loop(gloc, w, U, beta, 1e-10)
    
    # Select rho(omega) figure
    plt.figure(1)
    
    # Always choose same plot properties
    plt.gca().set_prop_cycle(None)
    
    # Shift vertically plot
    plt.axhline(-shift_g*i, color='k', lw=0.5)
    
    # Plot rho(omega)
    plt.plot(w, -shift_g*i + -gloc.imag)
    
    # Select sigma(omega) figure
    plt.figure(2)
    
    # Plot sigma(omega)
    plt.gca().set_prop_cycle(None)
    plt.axhline(-shift_s*i, color='k', lw=0.5)
    plt.plot(w, -shift_s*i + sigma_loc.imag, color='b', label='Imaginary part')
    plt.plot(w, -shift_s*i + sigma_loc.real, color='r', label='Real part')
    
    np.savetxt(".\data\Green_U=%s.txt"%U, np.transpose([w,gloc.real,gloc.imag]))
    np.savetxt(".\data\Sigma_U=%s.txt"%U, np.transpose([w,sigma_loc.real,sigma_loc.imag]))

# Save rho(omega)
plt.figure(1)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\rho$')
plt.xlim([-4, 4])
plt.ylim([-shift_g*len(urange), shift_g])
plt.yticks(-shift_g * np.arange(len(urange)), ['U=' + str(u) for u in urange])
plt.savefig(".\figures\DOS.png")

# save sigma(omega)
plt.figure(2)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\Sigma$')
plt.xlim([-4, 4])
plt.ylim([-shift_s*len(urange), shift_s])
plt.yticks(-shift_s * np.arange(len(urange)), ['U=' + str(u) for u in urange])
plt.legend()
plt.savefig(".\figures\sigma.png")

