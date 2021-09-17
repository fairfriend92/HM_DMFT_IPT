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
    # zeta  Frequencies in the Z plane
    # D     Bandwidth
    
    sqr = np.sqrt(zeta**2 - D**2)
    sqr = np.sign(sqr.imag) * sqr
    return 2 * (zeta - sqr) / D**2
    
def setup_figure(fig_idx, loop_idx, shift):
    plt.figure(fig_idx)
    
    # Always choose same plot properties
    plt.gca().set_prop_cycle(None)
    
    # Shift vertically plot
    plt.axhline(-shift*loop_idx, color='k', lw=0.5)
    
def save_figure(fig_idx, x_label, y_label, x_lim, shift):
    plt.figure(fig_idx)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_lim)
    plt.ylim([-shift * len(print_range), shift])
    plt.yticks(-shift * np.arange(len(print_range)), ['U=' + str(u) for u in print_range])
   
'''
MAIN LOOP
'''

dw = 0.01                               # Frequency increase
w = np.arange(-20, 20, dw)              # Frequencies  
gloc = semi_circle_hiltrans(w + 1e-3j)  # Local Green function
beta_range = [200, 100, 64]             # 1/temp
urange = np.arange(0.5, 5.0, 0.125)     # On site interaction term  
print_range = [0.5, 2.5, 3.5, 4.5]      # U values for which files should be saved
n_U_beta = []                           # Occupation number

shift_g = 2
shift_s = 40
shift_n = 2

for beta in beta_range:  
    n_beta = []
    print("beta="+str(beta))  
    j = 0
    for i, U in enumerate(urange):
        #print("U=%s"%(U))        
        
        gloc, sigma_loc = ss_dmft_loop(gloc, w, U, beta, 1e-10)
    
        # Electron concentration for temp 1/beta and energy w
        n_beta.append(np.sum(-gloc.imag/np.pi * fermi_dist(w, beta) * dw))
        
        if beta == 100 and U in print_range:
            # Plot rho(omega)
            setup_figure(1, j, shift_g)    
            plt.plot(w, -shift_g*j + -gloc.imag)
            
            # Plot sigma(omega)
            setup_figure(2, j, shift_s)
            plt.plot(w, -shift_s*j + sigma_loc.imag, color='b', label='Imaginary part')
            plt.plot(w, -shift_s*j + sigma_loc.real, color='r', label='Real part')  
            
            j = j+1
            
            np.savetxt("./data/Green_U=%s.txt"%U, np.transpose([w,gloc.real,gloc.imag]))
            np.savetxt("./data/Sigma_U=%s.txt"%U, np.transpose([w,sigma_loc.real,sigma_loc.imag]))            
    n_U_beta.append(n_beta)
    

# Save rho(omega)
save_figure(1, r'$\omega$', r'$\rho$', [-4, 4], shift_g)
plt.savefig("./figures/DOS.png")

# Save sigma(omega)
save_figure(2, r'$\omega$', r'$\Sigma$', [-4, 4], shift_s)
plt.legend()
plt.savefig("./figures/sigma.png")

# Save n(omega)
plt.figure(3)
plt.xlabel('U')
plt.ylabel('n')
for i in range(len(beta_range)):
    plt.plot(urange, n_U_beta[i], label=r'$\beta$='+str(beta_range[i]))
plt.legend()
plt.savefig("./figures/n.png")
