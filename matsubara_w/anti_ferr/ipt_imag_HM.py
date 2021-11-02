# -*- coding: utf-8 -*-
r"""
IPT Solver Single Band
======================

Within the iterative perturbative theory (IPT) the aim is to express the
self-energy of the impurity problem as

.. math:: \Sigma_\sigma(\tau) \approx U\mathcal{G}_\sigma^0(0)
    -U^2 \mathcal{G}_\sigma^0(\tau)
         \mathcal{G}_{\bar{\sigma}}^0(-\tau)
         \mathcal{G}_{\bar{\sigma}}^0(\tau)

The contribution of the Hartree-term is drooped because at half-filling
it cancels with the chemical potential. The next equalities are
obtained by using the anti-periodicity of the fermionic imaginary time
Green functions, and because of the half-filled case and particle hole
symmetry the last equality is fulfilled

.. math:: \mathcal{G}(-\tau) = -\mathcal{G}(-\tau+\beta)p = -\mathcal{G}(\tau)

As such for the single band paramagnetic case at half-filling the self
energy is estimated by

.. math:: \Sigma(\tau) \approx U^2 \mathcal{G}^0(\tau)^3

"""

from __future__ import division, absolute_import, print_function

#from numba import jit
from scipy.integrate import quad, simps
from scipy.optimize import fsolve
import numpy as np
from common import gt_fouriertrans, gw_invfouriertrans
import common as gf
import matplotlib.pylab as plt

def single_band_ipt_solver(u_int, g_0_iwn_up, g_0_iwn_dn, w_n, tau, n_up, n_dn, loop):
    r"""Given a Green function it returns a dressed one and the self-energy

    .. math:: \Sigma(\tau) \approx U^2 \mathcal{G}^0(\tau)^3

    Dyson eq.:
    
    .. math:: G = \mathcal{G}^0(i\omega_n)/(1 - \Sigma(i\omega_n)\mathcal{G}^0(i\omega_n))

    The Fourier transforms use as tail expansion of the atomic limit self-enegy

    .. math:: \Sigma(i\omega_n\rightarrow \infty) = \frac{U^2}{4(i\omega_n)}

    Parameters
    ----------
    u_int: float, local contact interaction
    g_0_iwn: complex 1D ndarray
        *bare* Green function, the Weiss field
    w_n: real 1D ndarray
        Matsubara frequencies
    tau: real 1D array
        Imaginary time points, not included edge point of :math:`\beta^-`
    """

    beta = tau[-1]
    mu = u_int / 2
    
    g_0_tau_up = gw_invfouriertrans(g_0_iwn_up, tau, w_n, [1., 0., 0.25])
    g_0_tb_up = [g_0_tau_up[len(tau)-1-t] for t in range(len(tau))]
    g_0_tau_dn = gw_invfouriertrans(g_0_iwn_dn, tau, w_n, [1., 0., 0.25])
    g_0_tb_dn = [g_0_tau_dn[len(tau)-1-t] for t in range(len(tau))]    
    
    n_0_up = 2/beta*np.sum(g_0_iwn_up) + 0.5
    n_0_dn = 2/beta*np.sum(g_0_iwn_dn) + 0.5
    
    plt.figure()
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$G_0(\tau)$')
    plt.plot(tau, g_0_tau_up.imag, label=r'$\sigma=\uparrow$ Im')
    plt.plot(tau, g_0_tau_up.real, label=r'$\sigma=\uparrow$ Re')
    plt.plot(tau, g_0_tau_dn.imag, label=r'$\sigma=\downarrow$ Im')
    plt.plot(tau, g_0_tau_dn.real, label=r'$\sigma=\downarrow$ Re')
    plt.legend()
    plt.savefig("./figures/Gtau_not_converged/Gtau_loop="+str(loop)+".png")
    plt.close()
    
    # IPT self-energy using G0 of quantum impurity
    sigma_tau_up = u_int**2 * g_0_tau_up * g_0_tb_dn * g_0_tau_dn
    sigma_tau_dn = u_int**2 * g_0_tau_dn * g_0_tb_up * g_0_tau_up
    sigma_iwn_up = gt_fouriertrans(sigma_tau_up, tau, w_n, [u_int**2 / 4., 0., 0.])
    sigma_iwn_dn = gt_fouriertrans(sigma_tau_dn, tau, w_n, [u_int**2 / 4., 0., 0.])
    
    A_up = n_dn * (1 - n_dn) / (n_0_dn * (1 - n_0_dn))
    A_dn = n_up * (1 - n_up) / (n_0_up * (1 - n_0_up))
    B_up = (u_int * (1 - n_dn) - mu) / (u_int**2 * n_0_dn * (1 - n_0_dn))
    B_dn = (u_int * (1 - n_up) - mu) / (u_int**2 * n_0_up * (1 - n_0_up))
    sigma_iwn_up = A_up * sigma_iwn_up/(1 - B_up * sigma_iwn_up)
    sigma_iwn_dn = A_dn * sigma_iwn_dn/(1 - B_dn * sigma_iwn_dn)
    
    plt.figure()
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\Sigma_0(\tau)$')
    plt.plot(tau, sigma_tau_up.imag, label=r'$\sigma=\uparrow$ Im')
    plt.plot(tau, sigma_tau_up.real, label=r'$\sigma=\uparrow$ Re')
    plt.plot(tau, sigma_tau_dn.imag, label=r'$\sigma=\downarrow$ Im')
    plt.plot(tau, sigma_tau_dn.real, label=r'$\sigma=\downarrow$ Re')
    plt.legend()
    plt.savefig("./figures/Sigmatau_not_converged/Sigmatau_loop="+str(loop)+".png")
    plt.close()
    
    # Soumen's implementation        
    zeta_up = w_n*1.j + mu - u_int*n_dn - sigma_iwn_up
    zeta_dn = w_n*1.j + mu - u_int*n_up - sigma_iwn_dn
    g_e_up = zeta_dn/np.add.outer(-e**2, (zeta_dn*zeta_up))
    g_e_dn = zeta_up/np.add.outer(-e**2, (zeta_dn*zeta_up))
    dos_de = (dos_e * de).reshape(-1, 1)
    g_iwn_up = (dos_de * g_e_up).sum(axis=0)
    g_iwn_dn = (dos_de * g_e_dn).sum(axis=0)
   
                
    # Dyson eq.
    #g_iwn_up = g_0_iwn_up / (1.0 - sigma_iwn_up * g_0_iwn_up)
    #g_iwn_dn = g_0_iwn_dn / (1.0 - sigma_iwn_dn * g_0_iwn_dn)    

    return g_iwn_up, g_iwn_dn, sigma_iwn_up, sigma_iwn_dn
    
def dmft_loop(u_int, t, g_iwn_up, g_iwn_dn, w_n, tau, mix=1, conv=1e-3):
    r"""Performs the paramagnetic(spin degenerate) self-consistent loop in a
    bethe lattice given the input

    Parameters
    ----------
    u_int : float
        Local interation strength
    t : float
        Hopping amplitude between bethe lattice nearest neighbors
    g_iwn : complex float ndarray
            Matsubara frequencies starting guess Green function
    tau : real float ndarray
            Imaginary time points. Only use the positive range
    mix : real :math:`\in [0, 1]`
            fraction of new solution for next input as bath Green function
    w_n : real float array
            fermionic matsubara frequencies. Only use the positive ones

    Returns
    -------
    tuple 2 complex ndarrays
            Interacting Green's function in matsubara frequencies
            Self energy
    """

    converged = False
    loops = 0
    iw_n = 1j * w_n
    beta = tau[-1]
    mu = u_int/2
    
    file = open("./data/dmft_loop.txt", "w") 
    file.write("n_up\tn_dn\tg_diff_up\tg_diff_dn\n")
    
    while not converged:
        # Initial condition for the magnetization
        m = 0.0 if loops == 0 else 0.0
        
        # Backup old G
        g_iwn_up_old = g_iwn_up.copy()
        g_iwn_dn_old = g_iwn_dn.copy()
        
        # Occupation numbers
        n_up = 2/beta*np.sum(g_iwn_up_old.real) + 0.5
        n_dn = 2/beta*np.sum(g_iwn_dn_old.real) + 0.5  
        
        # Non-interacting GF of quantum impurity  
        g_0_iwn_up = 1. / (iw_n + m - t**2 * g_iwn_up_old + mu - u_int*n_dn)
        g_0_iwn_dn = 1. / (iw_n - m - t**2 * g_iwn_dn_old + mu - u_int*n_up)
        
        # Impurity solver
        g_iwn_up, g_iwn_dn, sigma_iwn_up, sigma_iwn_dn = \
            single_band_ipt_solver(u_int, g_0_iwn_up, g_0_iwn_dn, w_n, tau, n_up, n_dn, loops)  
            
        # Clean for Half-fill
        #g_iwn_up.real = g_iwn_dn.real = 0.
        
        # Check convergence
        converged = np.allclose(g_iwn_up_old, g_iwn_up, conv) and np.allclose(g_iwn_dn_old, g_iwn_dn, conv)
        loops += 1
        if loops > 50:
            converged = True
        g_iwn_up = mix * g_iwn_up + (1 - mix) * g_iwn_up_old
        g_iwn_dn = mix * g_iwn_dn + (1 - mix) * g_iwn_dn_old
        
        # Write datafile 
        g_diff_up = np.abs(np.sum(g_iwn_up - g_iwn_up_old))
        g_diff_dn = np.abs(np.sum(g_iwn_dn - g_iwn_dn_old))
        file.write(str(n_up)+"\t"+str(n_dn)+"\t"+str(g_diff_up)+"\t"+str(g_diff_dn)+"\n")
        
    file.close()
    return g_iwn_up, g_iwn_dn, sigma_iwn_up, sigma_iwn_dn

# Parameters
t = 0.5     # Hopping
Nwn = 256  # Num of freq: 1024 is recommended
U_max = 4.5 # Electron interaction

# Ranges 
U_list = np.arange(2.0, U_max, 0.5)   # Range of interaction strength  
U_print = np.arange(2.0, U_max, 0.5)  # U values for which observables are printed
beta_list = np.arange(5, 100, 10)     # Inverse of temperature
beta_print = np.arange(5, 100, 10)    # Beta values for which observables are printed   
dw = 0.01                             # Real freq differential
w = np.arange(-15, 15, dw)            # Real freq
de = 2*t/1000                         # Energy differential
e = np.arange(-2*t, 2*t, de)          # Energy
dos_e = gf.bethe_dos(t, e)            # Bethe lattice DOS

#U_list = U_print = np.array([0.5, 1., 1.5, 2., 2.5])
beta_list = beta_print = [100.] #np.array([1/T for T in np.arange(0.01, 0.16, 0.01)])

# Hysteresis
hyst = False    
if (hyst):
    U_list = np.append(U_list, np.arange(U_max, 0.375, -0.125))
    U_print = np.append(U_print, U_print[::-1])
    
# Observables
tau_U = []
dos_U = []
n_U = []
d_U = []
Ekin_U = []
Z_U = []
phase_U = []
Gwn_U_up = []
Gwn_U_dn = []
Gtau_U_up = []
Gtau_U_dn = []

# Main loop
for beta in beta_list:
    print("beta="+f'{beta:.3f}')

    # Generate Matsubara freq
    tau, wn = gf. tau_wn_setup(beta, Nwn) 
    
    # Seed green function
    G_iwn_up = gf.greenF(wn, 0, 0, 2*t)
    G_iwn_dn = gf.greenF(wn, 0, 0, 2*t)    
    plt.figure()
    plt.xlabel(r'$\omega_n$')
    plt.ylabel(r'$G(\omega_n)$')
    plt.plot(wn, G_iwn_up.imag, label=r'$\sigma=\uparrow$ Im')
    plt.plot(wn, G_iwn_up.real, label=r'$\sigma=\uparrow$ Re')
    plt.plot(wn, G_iwn_dn.imag, label=r'$\sigma=\downarrow$ Im')
    plt.plot(wn, G_iwn_dn.real, label=r'$\sigma=\downarrow$ Re')
    plt.legend()
    plt.savefig("./figures/Gseed.png")
    plt.close()
    
    # Index of zero frequency
    w0_idx = int(len(w)/2)
    
    dos_beta = []
    n_beta = []
    d_beta = []
    Ekin_beta = []
    Z_beta = []
    phase_beta = []
    Gwn_beta_up = []
    Gwn_beta_dn = []
    Gtau_beta_up = []
    Gtau_beta_dn = []

    for U in U_list:
        G_iwn_up, G_iwn_dn, Sig_iwn_up, Sig_iwn_dn = \
            dmft_loop(U, t, G_iwn_up, G_iwn_dn, wn, tau, mix=1, conv=1e-3)
        
        G_iwn = G_iwn_up
        Sig_iwn = Sig_iwn_up
        
        # Imaginary time Green function
        G_tau_up = gw_invfouriertrans(G_iwn_up, tau, wn, [1., 0., 0.25])
        G_tau_dn = gw_invfouriertrans(G_iwn_dn, tau, wn, [1., 0., 0.25])
        
        # Analytic continuation using Pade
        #g_w = gf.pade_continuation(G_iwn, wn, w, w_set=None)
        #sig_w = gf.pade_continuation(Sig_iwn, wn, w, w_set=None)
        
        # Phase of material, -1 for metallic, 1 for insulating
        #phase = -1 if -g_w[w0_idx].imag > 1e-2 else 1
        #phase_beta.append(phase)          
                
        if U in U_print and beta in beta_print:
            # Save Green functions
            Gwn_beta_up.append(G_iwn_up)
            Gwn_beta_dn.append(G_iwn_dn)
            Gtau_beta_up.append(G_tau_up)
            Gtau_beta_dn.append(G_tau_dn)
            
            print("T="+f'{1/beta:.3f}'+"\tU="+f'{U:.3}'+"\tG_w0.im="+f'{G_iwn_up[0].imag:.3f}')
            
            # DOS
            #dos_beta.append(-g_w.imag/np.pi)
            
            # Electron concentration for temp 1/beta and energy w
            #n = np.sum(-g_w.imag/np.pi * gf.fermi_dist(w, beta) * dw)
            n = np.sum(G_iwn.real) + 0.5
            n_beta.append(n)
            
            # Double occupancy
            d = n**2 + 1/(U*beta)*np.sum(G_iwn*Sig_iwn)
            d_beta.append(d.real)
            
            # Kinetic energy
            Ekin = 0
            # Sum over Matsubara freq
            for n in range(Nwn):
                # Integral in epsilon
                Ekin += 2/beta * np.sum(de * e * dos_e * gf.g_k_w(e, wn[n], Sig_iwn[n], mu=0.0))
            Ekin_beta.append(Ekin.real)
            
            # Quasi-particle weight
            #dSig = (sig_w[w0_idx+1].real-sig_w[w0_idx].real)/dw
            #Z_beta.append(1/(1-dSig))
    
    if beta in beta_print:
        tau_U.append(tau)
        dos_U.append(dos_beta)
        n_U.append(n_beta)
        d_U.append(d_beta)
        Ekin_U.append(Ekin_beta)
        Z_U.append(Z_beta)
        phase_U.append(phase_beta)
        Gwn_U_up.append(Gwn_beta_up)
        Gwn_U_dn.append(Gwn_beta_dn)
        Gtau_U_up.append(Gtau_beta_up)
        Gtau_U_dn.append(Gtau_beta_dn)
    
# Print DOS
'''
for i in range(len(beta_print)):
    dos = dos_U[i]
    beta = beta_print[i]
    plots = int(len(U_print)/2) if hyst else len(U_print)
    fig, axs = plt.subplots(plots, sharex=True, sharey=True)
    for j in range(plots):
        axs[j].set(xlabel=r'$\omega$')
        axs[j].plot(w, dos[j])   
    fig.supylabel(r'$\rho(\omega)$')    
    plt.title(r'$\beta=$'+f'{beta:.3}')
    plt.savefig("./figures/dos_beta="+f'{beta:.3}'+".png")
    plt.close()
'''

# Print Green functions
for i in range(len(beta_print)):
    beta = beta_print[i]
    tau = tau_U[i]
    Gwn_up = Gwn_U_up[i]
    Gwn_dn = Gwn_U_dn[i]
    Gtau_up = Gtau_U_up[i]
    Gtau_dn = Gtau_U_dn[i]
    for j in range(len(U_print)):
        U = U_print[j]
        
        # Matsubara Green function
        plt.figure()
        plt.xlabel(r'$\omega_n$')
        plt.ylabel(r'$G(\omega_n)$')
        plt.plot(wn, Gwn_up[j].imag, label=r'$\sigma=\uparrow$ Im')
        plt.plot(wn, Gwn_up[j].real, label=r'$\sigma=\uparrow$ Re')
        plt.plot(wn, Gwn_dn[j].imag, label=r'$\sigma=\downarrow$ Im')
        plt.plot(wn, Gwn_dn[j].real, label=r'$\sigma=\downarrow$ Re')
        plt.legend()
        plt.savefig("./figures/Gwn/Gwn_beta="+f'{beta:.3}'+"_U="+f'{U:.3}'+".png")
        plt.close()
        
        # Imaginary time Green function
        plt.figure()
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$G(\tau)$')
        plt.plot(tau, Gtau_up[j].imag, label=r'$\sigma=\uparrow$ Im')
        plt.plot(tau, Gtau_up[j].real, label=r'$\sigma=\uparrow$ Re')
        plt.plot(tau, Gtau_dn[j].imag, label=r'$\sigma=\downarrow$ Im')
        plt.plot(tau, Gtau_dn[j].real, label=r'$\sigma=\downarrow$ Re')
        plt.legend()
        plt.savefig("./figures/Gtau/Gtau_beta="+f'{beta:.3}'+"_U="+f'{U:.3}'+".png")
        plt.close()

# Print zero-freq Matsubara Green function
for i in range(len(beta_print)):
    beta = beta_print[i]
    Gwn = Gwn_U_up[i]
    Gw0 = []
    for g in Gwn:
        Gw0.append(g[0].imag)        
    plt.figure()
    plt.xlabel(r'$U$')
    plt.ylabel(r'$G(\omega_0)$')
    plt.plot(U_print, Gw0)
    plt.savefig("./figures/Gw0/Gw0_beta="+f'{beta:.3}'+".png")
    plt.close()

# Print n
plt.figure()
for i in range(len(beta_print)):
    n = n_U[i]
    beta = beta_print[i]    
    plt.xlabel('U')
    plt.ylabel('n')
    plt.plot(U_print, n, label='beta='+f'{beta:.3}')
    plt.legend()
plt.savefig("./figures/n.png")


# Print d
plt.figure()
for i in range(len(beta_print)):
    d = d_U[i]
    beta = beta_print[i]    
    plt.xlabel('U')
    plt.ylabel('d')
    plt.plot(U_print, d, label='beta='+f'{beta:.3}')
    plt.legend()
plt.savefig("./figures/d.png")

# Print kinetic energy
plt.figure()
for i in range(len(beta_print)):
    Ekin = Ekin_U[i]
    beta = beta_print[i]
    plt.xlabel('U')
    plt.ylabel(r'$E_K$')
    plt.xlim(2, 3.5)
    plt.ylim(-0.5, 0)
    plt.plot(U_print, Ekin, label='beta='+f'{beta:.3}')
    plt.legend()
plt.savefig("./figures/Ekin.png")

# Print quasi-particle weight
'''
plt.figure()
for i in range(len(beta_print)):
    Z = Z_U[i]
    beta = beta_print[i]
    plt.xlabel('U')
    plt.ylabel('Z')
    plt.plot(U_print, Z, label='beta='+f'{beta:.3}')
    plt.legend()
plt.savefig("./figures/Z.png")
'''

def get_phase(U, T, val):
    # G for different T, U values
    Gwn = np.flipud(Gwn_U_up) # Increasing temp order
    if np.abs(Gwn[T][U][0].imag) < val:
        return -1             # Metallic phase
    else:
        return 1              # Insulating phase

# Print phase diagram
if (len(beta_list) > 1):
    plt.figure()
    plt.xlabel('U')
    plt.ylabel('T')
    T_list = [1/beta for beta in beta_list]    # Convert from beta to temp
    T_list = np.flipud(T_list)                 # Sort in increasing order
    T_trans = [0.]                             # Transition temperature
       
    for i in range(len(U_print)):
        for j in range(len(T_list)-1):
            if get_phase(i, j, 0.15) != get_phase(i, j+1, 0.15):
                T_trans.append(T_list[j])
                break
            elif j == len(T_list)-2:    
                T_trans.append(-1)   # No transition point
                
    U_print = np.insert(U_print, 0, 0)  # Insert starting U
    
    # Restrict curve to valid transition points
    T_arr = np.array(T_trans)
    U_arr = np.array(U_print) 
    U_mask = U_arr[T_arr > -1]
    T_mask = T_arr[T_arr > -1]
   
    plt.xlim(left = 0, right = U_max)
    plt.plot(U_mask, T_mask)
            
'''
phase_U = np.flipud(phase_U)
phase_diag = []
if hyst:
    for phase_beta in phase_U:
        inc_U = phase_beta[:int(len(U_list)/2)]
        dec_U = phase_beta[int(len(U_list)/2)+1:]
        dec_U = np.flipud(dec_U)
        final = inc_U + dec_U
        phase_diag.append(final)       
im_edges = [U_list[0], U_max,
            T_list[0], T_list[len(T_list)-1]]
plt.imshow(phase_diag, interpolation='none', extent=im_edges, aspect='auto')
'''

plt.savefig("./figures/phase_diag.png")
