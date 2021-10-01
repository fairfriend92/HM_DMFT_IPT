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
    # Convolution A^-(-w_range) * App
    Appp = signal.fftconvolve(Ap, App, mode='same')
    return -np.pi * U**2 * (Appp + Appp[::-1])

'''
DMFT Loop for the single band Hubbard Model at Half-Filling
'''
def ss_dmft_loop(gloc, w_range, u_int, beta, conv):
    # gloc  Local Green's function to use as seed   (complex 1D ndarray) 
    # sigma Self-energy                             (complex 1D ndarray)
    # w_range     Real frequency points             (real 1D ndarray) 
    # u_int On site interaction                     (float)
    # beta  1/T                                     (float)
    # conv  Convergence criteria                    (float)
  
    dw = w_range[1] - w_range[0]
    eta = 2j * dw
    nf = fermi_dist(w_range, beta)

    converged = False
    while not converged:
        gloc_old = gloc.copy()
        
        # Self-consistency
        g0 = 1 / (w_range + eta - .25 * gloc)
        
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
        gloc = semi_circle_hiltrans(w_range - sigma)
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
    plt.title(r'$\beta=$'+str(beta))
    
'''
Fourier transform using spectral function
'''
def f_tau(f, tau, beta):    
    rho_w = - 1 / np.pi * f.imag
    if -beta < tau and tau < 0:
        tau = tau+beta
    return np.sum(rho_w * np.exp(-tau*w_range) / (1 + np.exp(-beta*w_range)) * dw)   
   
'''
MAIN LOOP
'''
dw = 0.01                                               # Frequency increase
w_range = np.arange(-15, 15, dw, dtype=np.float128)     # Frequencies  
#beta_range = np.arange(32, 164, 16, dtype=np.float128)  # 1/temp
beta_range = [100]
U_range = np.arange(0.5, 5.0, 0.125)                    # On site interaction term  
print_range = [0.5, 2.5, 3.5, 4.5]                      # U values for which files should be saved

n_U_beta = []   # Occupation numbers
nn_U_beta = []  # Double occupancy
m_U_beta = []   # Effective mass
phase_diag = [] # Phase diagram

# Find index of zero frequency
w0_idx = 0 
for i in range(len(w_range)):
    w = w_range[i]
    if w <= dw and w >= -dw:
        w0_idx = i; 

# Vertical shift for when figure has multiple plots
shift_g = 2                 
shift_s = 40
shift_n = 2
shift_gloc = 0.5

for beta in beta_range:  
    print("beta=" + str(beta))

    # Seed local Green function
    gloc = semi_circle_hiltrans(w_range + 1e-3j) 
    
    # Arrays for fixed beta value
    n_beta = [] 
    nn_beta = []
    m_beta = []
    phase_beta = []
    
    j = 0
    for i, U in enumerate(U_range):     
        gloc, sigma_loc = ss_dmft_loop(gloc, w_range, U, beta, 1e-10)
        
        # Electron concentration for temp 1/beta and energy w_range
        n = np.sum(-gloc.imag/np.pi * fermi_dist(w_range, beta) * dw)
        n_beta.append(n)
        
        # G(tau)        
        dtau = beta/100
        tau_range = np.arange(0, beta+1, dtau, dtype=np.float128)
        nn = 0
        gloc_tau = []
        for tau in tau_range:
            g = f_tau(gloc, tau, beta)
            gloc_tau.append(g)
            '''
            s = f_tau(sigma_loc, beta-tau, beta)            
            nn += -1/U * g * s * dtau
            '''
        # Double occupancy    
        gsigma = gloc.real*sigma_loc.imag + gloc.imag*(sigma_loc.real + U*n)
        nn_beta.append(-1/np.pi*np.sum(fermi_dist(w_range, beta)*gsigma)*dw)
        
        # Effective mass
        dSigma = (sigma_loc[w0_idx+1].real-sigma_loc[w0_idx].real)/dw
        m_beta.append(1/(1-dSigma))
    
        # Phase of material, 0 for metallic, 1 for insulating
        phase = 0 if -gloc[w0_idx].imag > 0.05 else 1
        phase_beta.append(phase)        
        
        if U in print_range:                   
            # Plot rho(omega)            
            setup_figure(1+beta, j, shift_g)  
            plt.plot(w_range, -shift_g*j + -gloc.imag)
            
            # Plot sigma(omega)
            setup_figure(2+beta, j, shift_s)
            plt.plot(w_range, -shift_s*j + sigma_loc.imag, color='b', label='Imaginary part')
            plt.plot(w_range, -shift_s*j + sigma_loc.real, color='r', label='Real part') 

            # Plot gloc(tau)
            setup_figure(3+beta, j, shift_gloc) 
            plt.plot(tau_range, [-shift_gloc*j + g for g in gloc_tau])
            
            j = j+1
            
            np.savetxt("./data/w_Greal_Gimag_U="+str(U)+"_beta="+str(beta)+".txt", 
                       np.transpose([w_range,gloc.real,gloc.imag]))
            np.savetxt("./data/w_Sreal_Simag_U="+str(U)+"_beta="+str(beta)+".txt", 
                       np.transpose([w_range,sigma_loc.real,sigma_loc.imag])) 
            np.savetxt("./data/Tau_Gtau_U="+str(U)+"_beta="+str(beta)+".txt", 
                       np.transpose([tau_range, gloc_tau])) 
    n_U_beta.append(n_beta)
    nn_U_beta.append(nn_beta)
    m_U_beta.append(m_beta)
    phase_diag.append(phase_beta)
    
''' 
Save figures 
'''
for beta in beta_range:
    # Save rho(omega)
    save_figure(1+beta, r'$\omega$', r'$\rho$', [-4, 4], shift_g)
    plt.savefig("./figures/DOS_beta="+str(beta)+".png")

    # Save sigma(omega)
    save_figure(2+beta, r'$\omega$', r'$\Sigma$', [-4, 4], shift_s)
    plt.legend()
    plt.savefig("./figures/sigma_beta="+str(beta)+".png")
    
    # Save gloc(tau)
    save_figure(3+beta, r'$\tau$', r'$G$', [0, beta], shift_gloc)
    plt.savefig("./figures/gloc_beta="+str(beta)+".png")

# Save double occupancy
plt.figure(3)
plt.xlabel('U')
plt.ylabel('nn')
#plt.xlim([2.8, 4])
#plt.ylim([-3, -1])
for i in range(len(beta_range)):
    plt.plot(U_range, nn_U_beta[i], label=r'$\beta$='+str(beta_range[i]))
plt.legend()
plt.savefig("./figures/nn.png")
plt.close(3)

# Save occupation number
plt.figure(4)
plt.xlabel('U')
plt.ylabel('n')
for i in range(len(beta_range)):
    plt.plot(U_range, n_U_beta[i], label=r'$\beta$='+str(beta_range[i]))
plt.legend()
plt.savefig("./figures/n.png")
plt.close(4)

# Save effective mass
plt.figure(5)
plt.xlabel('U')
plt.ylabel('m*')
for i in range(len(beta_range)):
    plt.plot(U_range, m_U_beta[i], label=r'$\beta$='+str(beta_range[i]))
plt.legend()
plt.savefig("./figures/m_eff.png")
plt.close(5)

# Save phase diagram
plt.figure(6)
plt.xlabel('U')
plt.ylabel('T')
T_range = [1/beta for beta in beta_range]   # Convert from beta to temp
T_range = np.flipud(T_range)                # Sort in increasing order
phase_diag = np.flipud(phase_diag)
im_edges = [U_range[0], U_range[len(U_range)-1],
            T_range[0], T_range[len(T_range)-1]]
plt.imshow(phase_diag, interpolation='none', extent=im_edges, aspect='auto')
plt.savefig("./figures/phase_diag.png")

