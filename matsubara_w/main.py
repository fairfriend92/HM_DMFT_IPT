import numpy as np
import matplotlib.pylab as plt
import dmft
from constants import *
from green_func import ift   
import print_func as print_f
from pade import my_pade

''' Main loop '''

test = [[]]
tau_U = []
dos_U = []
n_U = []
d_U = []
ekin_U = []
Z_U = []
phase_U = []
g_wn_U_up = []
g_wn_U_dn = []
g_tau_U_up = []
g_tau_U_dn = []

for beta in beta_list:  
    # Generate Matsubara freq 
    wn = np.pi * (1 + 2 * np.arange(-N, N, dtype=np.double)) / beta

    # Generate imaginary time 
    dtau = beta/(2*N)   # tau has to be twice as dense as wn...
                        # ...when considering negative freq
    tau = np.arange(dtau/2., beta, dtau, dtype=np.double)
     
    # Seed green function
    g_wn_up = -2.j / (wn + np.sign(wn) * np.sqrt(wn**2 + D**2))
               #1/(1.j*wn + 1.j*D*np.sign(wn)) 
               
    g_wn_dn = g_wn_up    
    print_f.generic(wn, g_wn_up, g_wn_dn, 
                    r'$\omega_n$', r'$g(\omega_n)$', 
                    "./figures/g_seed.pdf")
    
    # Index of zero frequency
    w0_idx = int(len(w)/2)
    
    dos_beta = []
    n_beta = []
    d_beta = []
    e_kin_beta = []
    Z_beta = []
    phase_beta = []
    g_wn_beta_up = []
    g_wn_beta_dn = []
    g_tau_beta_up = []
    g_tau_beta_dn = []

    for U in U_list:
        g_wn_up, g_wn_dn, sig_wn_up, sig_wn_dn = \
            dmft.loop(U, t, g_wn_up, g_wn_dn, wn, tau, beta, 
                      mix=1., conv=1e-3, max_loops=50, m_start=0.0)
        
        g_wn = g_wn_up
        sig_wn = sig_wn_up
        
        # Imaginary time Green function
        g_tau_up = ift(wn, g_wn_up, tau, beta)
        g_tau_dn = ift(wn, g_wn_dn, tau, beta)
        
        # Analytic continuation using Pade
        #print(wn)
        if do_pade:
            g_w = my_pade(g_wn, w, wn)
            sig_w = my_pade(sig_wn, w, wn)
                
        if U in U_print and beta in beta_print:           
            # Save Green functions
            g_wn_beta_up.append(g_wn_up)
            g_wn_beta_dn.append(g_wn_dn)
            g_tau_beta_up.append(g_tau_up)
            g_tau_beta_dn.append(g_tau_dn)
            
            print("T="+f'{1/beta:.3f}'+"\tU="+f'{U:.3}')
                        
            # DOS
            if do_pade:
                dos_beta.append(-g_w.imag/np.pi)
            
            # Electron concentration for temp 1/beta and energy w
            n = np.sum(g_wn.real) + 0.5
            #print("n="+f'{n:.5f}')
            n_beta.append(n)
            
            # Double occupancy
            d = n**2 + 1/(U*beta)*np.sum(g_wn*sig_wn)
            d_beta.append(d.real)
            
            # Kinetic energy
            e_kin = 0.
            # Sum over Matsubara freq
            for w_n, sig_n in zip(wn[N:], sig_wn[N:]):
                # Integral in epsilon
                mu = 0.
                g_k_wn = 1./(1.j*w_n + mu - e - sig_n)
                e_kin += 2./beta * np.sum(de*e*dos_e*g_k_wn)
            #print("E_kin.real="+f'{e_kin.real:.5f}')
            e_kin_beta.append(e_kin.real)
            
            # Quasi-particle weight
            if do_pade:
                dSig = (sig_w[w0_idx+1].real-sig_w[w0_idx].real)/dw
                Z_beta.append(1/(1-dSig))
    
    if beta in beta_print:
        tau_U.append(tau)
        dos_U.append(dos_beta)
        n_U.append(n_beta)
        d_U.append(d_beta)
        ekin_U.append(e_kin_beta)
        Z_U.append(Z_beta)
        phase_U.append(phase_beta)
        g_wn_U_up.append(g_wn_beta_up)
        g_wn_U_dn.append(g_wn_beta_dn)
        g_tau_U_up.append(g_tau_beta_up)
        g_tau_U_dn.append(g_tau_beta_dn)
        
''' Printing functions '''      

print_f.green_func(beta_print, tau_U, \
                    g_wn_U_up, g_wn_U_dn, g_tau_U_up, g_tau_U_dn, \
                    U_print, hyst, wn)
print_f.gf_iw0(beta_print, g_wn_U_up, U_print)
print_f.n(beta_print, n_U, U_print)
print_f.d(beta_print, d_U, U_print)

print_f.e_kin(beta_print, ekin_U, U_print)
print_f.phase(beta_list, U_print, g_wn_U_up)

if do_pade:
    print_f.dos(beta_print, w, dos_U, U_print, hyst)
    print_f.Z(beta_print, Z_U, U_print)
