import numpy as np
import matplotlib.pylab as plt
import dmft
import green_func as green_f
import print_func as print_f
from pade import pade_continuation

''' Variables '''

# Parameters
t = 0.5         # Hopping
D = 2 * t       # Half-bandwidth
num_freq = 256  # Num of freq: 1024 is recommended
hyst = False    # If true loop for decreasing U   
do_pade = False # If true use Pade's continuation 

# Electron interaction
U_min = 2.0
dU = 0.5
U_max = 2.5 
U_list = np.arange(U_min, U_max, dU)    
U_print = np.arange(U_min, U_max, dU)   
if (hyst):
    U_list = np.append(U_list, np.arange(U_max-dU, U_min-dU, -dU))
    U_print = np.append(U_print, U_print[::-1])

# Inverse of temperature 
beta_list = [128.]        
beta_print = beta_list     

# Real frequency
dw = 0.01                             
w = np.arange(-15, 15, dw)           

# Energy
de = 2*t/256                         
e = np.arange(-2*t, 2*t, de)         
dos_e = 2*np.sqrt(D**2 - e**2) / (np.pi * D**2) # Bethe lattice DOS
 
# Observables
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

''' Main loop '''

for beta in beta_list:
    # Generate Matsubara freq
    wn = np.pi * (1 + 2 * np.arange(-num_freq, num_freq)) / beta
    dtau = beta/num_freq
    tau = np.arange(dtau, beta, dtau)

    # Seed green function
    g_wn_up = green_f.bethe_gf(wn, 0.0, 0.0, 2*t) 
    g_wn_dn = g_wn_up    
    print_f.generic(wn, g_wn_up, g_wn_dn, 
                    r'$\omega_n$', r'$g(\omega_n)$', 
                    "./figures/g_seed.png")
    
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
        g_wn_up, g_wn_dn, sig_wn_up, Sig_iwn_dn = \
            dmft.loop(U, t, g_wn_up, g_wn_dn, wn, tau, beta,
                      mix=1., conv=1e-3, max_loops=50, m_start=0.)
        
        g_wn = g_wn_up
        sig_wn = sig_wn_up
        
        # Imaginary time Green function
        g_tau_up = green_f.ift(wn, g_wn_up, tau, beta)
        g_tau_dn = green_f.ift(wn, g_wn_dn, tau, beta)
        
        # Analytic continuation using Pade
        if do_pade:
            g_w = pade_continuation(g_wn, wn, w, w_set=None)
            sig_w = pade_continuation(sig_wn, wn, w, w_set=None)
                
        if U in U_print and beta in beta_print:
            # Save Green functions
            g_wn_beta_up.append(g_wn_up)
            g_wn_beta_dn.append(g_wn_dn)
            g_tau_beta_up.append(g_tau_up)
            g_tau_beta_dn.append(g_tau_dn)
            
            print("T="+f'{1/beta:.3f}'+"\tU="+f'{U:.3}'+"\tg_w0.im="+f'{g_wn_up[0].imag:.3f}')
                        
            # DOS
            if do_pade:
                dos_beta.append(-g_w.imag/np.pi)
            
            # Electron concentration for temp 1/beta and energy w
            n = np.sum(g_wn.real) + 0.5
            n_beta.append(n)
            
            # Double occupancy
            d = n**2 + 1/(U*beta)*np.sum(g_wn*sig_wn)
            d_beta.append(d.real)
            
            # Kinetic energy
            e_kin = 0
            # Sum over Matsubara freq
            for n in range(num_freq):
                # Integral in epsilon
                mu = 0.0
                g_k_wn = 1/(1.j*wn[n] + mu - e - sig_wn[n])
                e_kin += 2/beta * np.sum(de * e * dos_e * g_k_wn)
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
    print_f.dos(beta_print, dos_U, U_print, hyst)
    print_f.Z(beta_print, Z_U, U_print)
