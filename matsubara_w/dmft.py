import numpy as np
import matplotlib.pylab as plt
from green_func import ft as ft    
from green_func import ift as ift   
import print_func as print_f

def save_g_sigma(wn, tau, g_0_wn_up, g_0_wn_dn, g_0_tau_up, g_0_tau_dn,
                 sigma_wn_up, sigma_wn_dn, 
                 sigma_tau_up, sigma_tau_dn, 
                 g_wn_up, g_wn_dn, loop):
    # Print g_0_wn
    print_f.generic(wn, g_0_wn_up, g_0_wn_dn, 
                    r'$\omega_n$', r'$G_0(\omega_n)$', 
                    "./figures/not_converged/g_0_wn_loop="+str(loop)+".pdf")  
    
    # Write g_0_wn
    file = open("./data/g_0_wn_up_loop="+str(loop)+".txt", "w") 
    file.write("wn\tg_0_wn_up\n")
    for w, g, in zip(wn, g_wn_up):
        file.write(str(w) + "\t" + str(g) + "\n")
    file.close()
                    
    # Print g_0_tau                
    print_f.generic(tau, g_0_tau_up, g_0_tau_dn, 
                    r'$\tau$', r'$G_0(\tau)$', 
                    "./figures/not_converged/g_0_tau_loop="+str(loop)+".pdf") 
    
    # Write g_0_tau      
    file = open("./data/g_0_tau_up_loop="+str(loop)+".txt", "w") 
    file.write("tau\tg_0_tau_up\n")
    for t, g, in zip(tau, g_0_tau_up):
        file.write(str(t) + "\t" + str(g) + "\n")
    file.close()
    
    # Print sigma_wn 
    print_f.generic(wn, sigma_wn_up, sigma_wn_dn, 
                    r'$\omega_n$', r'$\Sigma(\omega_n)$', 
                    "./figures/not_converged/sig_wn_loop="+str(loop)+".pdf")
    
    # Print sigma_tau
    print_f.generic(tau, sigma_tau_up, sigma_tau_dn, 
                    r'$\tau$', r'$\Sigma(\tau)$', 
                    "./figures/not_converged/sig_tau_loop="+str(loop)+".pdf")
    
    # Print g_wn
    print_f.generic(wn, g_wn_up, g_wn_dn, 
                    r'$\omega_n$', r'$G(\omega_n)$', 
                    "./figures/not_converged/g_wn_loop="+str(loop)+".pdf")

def ipt_solver_para_mag(beta, U, g_0_wn, wn, tau, loop):     
    g_0_tau = ift(wn, g_0_wn, tau, beta, a=1.)
    
    # IPT self-energy using G0 of quantum impurity
    sigma_tau = U**2 * g_0_tau**3 
    sigma_wn = ft(wn, sigma_tau, tau, beta)
    
    # Dyson eq.
    g_wn = g_0_wn / (1.0 - sigma_wn * g_0_wn)
    
    # Print G and self-energy
    save_g_sigma(wn, tau, 
                 g_0_wn, g_0_wn, g_0_tau, g_0_tau,
                 sigma_wn, sigma_wn,
                 sigma_tau, sigma_tau, 
                 g_wn, g_wn, loop)

    return g_wn, sigma_wn

def ipt_solver_anti_ferr(beta, U, g_0_wn_up, g_0_wn_dn, wn, tau, 
                         n_up, n_dn, loop):    
    mu = U / 2
       
    g_0_tau_up = ift(wn, g_0_wn_up, tau, beta, a=1.)
    g_0_tb_up = np.array([g_0_tau_up[len(tau)-1-t] for t in range(len(tau))])
    g_0_tau_dn = ift(wn, g_0_wn_dn, tau, beta, a=1.)
    g_0_tb_dn = np.array([g_0_tau_dn[len(tau)-1-t] for t in range(len(tau))])   
    
    # IPT self-energy using G0 of quantum impurity
    sigma_tau_up = U**2 * g_0_tau_up* g_0_tb_dn * g_0_tau_dn
    sigma_tau_dn = U**2 * g_0_tau_dn* g_0_tb_up * g_0_tau_up
    sigma_wn_up = ft(wn, sigma_tau_up, tau)
    sigma_wn_dn = ft(wn, sigma_tau_dn, tau)
    
    # Soumen's implementation  
    n_0_up = 2/beta*np.sum(g_0_wn_up) + 0.5
    n_0_dn = 2/beta*np.sum(g_0_wn_dn) + 0.5
    A_up = n_dn * (1 - n_dn) / (n_0_dn * (1 - n_0_dn))
    A_dn = n_up * (1 - n_up) / (n_0_up * (1 - n_0_up))
    B_up = (U * (1 - n_dn) - mu) / (U**2 * n_0_dn * (1 - n_0_dn))
    B_dn = (U * (1 - n_up) - mu) / (U**2 * n_0_up * (1 - n_0_up))
    sigma_wn_up = A_up * sigma_wn_up/(1 - B_up * sigma_wn_up)
    sigma_wn_dn = A_dn * sigma_wn_dn/(1 - B_dn * sigma_wn_dn)

    zeta_up = wn*1.j + mu - U*n_dn - sigma_wn_up
    zeta_dn = wn*1.j + mu - U*n_up - sigma_wn_dn
    g_e_up = zeta_dn/np.add.outer(-e**2, (zeta_dn*zeta_up))
    g_e_dn = zeta_up/np.add.outer(-e**2, (zeta_dn*zeta_up))
    dos_de = (dos_e * de).reshape(-1, 1)
    g_wn_up = (dos_de * g_e_up).sum(axis=0)
    g_wn_dn = (dos_de * g_e_dn).sum(axis=0)
  
    # Print G and self-energy
    save_g_sigma(wn, tau, 
                 g_0_wn_up, g_0_wn_dn, g_0_tau_up, g_0_tau_dn,
                 sigma_wn_up, sigma_wn_dn, 
                 sigma_tau_up, sigma_tau_dn, 
                 g_wn_up, g_wn_dn, loop)

    return g_wn_up, g_wn_dn, sigma_wn_up, sigma_wn_dn
    
def loop(U, t, g_wn_up, g_wn_dn, wn, tau, beta, 
         mix=1, conv=1e-3, max_loops=50, m_start=0.):
    converged = False
    loops = 0
    mu = U/2
    
    file = open("./data/dmft_loop_beta="+f'{beta:.3}'+"_U="+f'{U:.3}'+".txt", "w") 
    file.write("n_up\tn_dn\tg_diff_up\tg_diff_dn\n")
    
    while not converged:
        # Initial condition for the magnetization
        m = m_start if loops == 0 else 0.0
        
        # Backup old g
        g_wn_up_old = g_wn_up.copy()
        g_wn_dn_old = g_wn_dn.copy()
        
        # Occupation numbers
        n_up = 2/beta*np.sum(g_wn_up_old.real) + 0.5
        n_dn = 2/beta*np.sum(g_wn_dn_old.real) + 0.5 
       
        # Non-interacting GF of quantum impurity  
        g_0_wn_up = 1. / (1.j*wn + m - t**2 * g_wn_up_old + mu - U*n_dn)
        g_0_wn_dn = 1. / (1.j*wn - m - t**2 * g_wn_dn_old + mu - U*n_up)
        
        # Impurity solver
        if (m_start != 0.):
            g_wn_up, \
            g_wn_dn, \
            sigma_wn_up, \
            sigma_wn_dn = ipt_solver_anti_ferr(beta, U, g_0_wn_up, g_0_wn_dn, wn, 
                                               tau, n_up, n_dn, loops)  
        else:
            g_wn_up, \
            sigma_wn_up = ipt_solver_para_mag(beta, U, g_0_wn_up, wn, 
                                              tau, loops)  
            g_wn_dn = g_wn_up
            sigma_wn_dn = sigma_wn_up
       
        # Check convergence
        converged = np.allclose(g_wn_up_old, g_wn_up, conv) and np.allclose(g_wn_dn_old, g_wn_dn, conv)
        loops += 1
        if loops > max_loops:
            converged = True
        g_wn_up = mix * g_wn_up + (1 - mix) * g_wn_up_old
        g_wn_dn = mix * g_wn_dn + (1 - mix) * g_wn_dn_old
        
        # Write datafile 
        g_diff_up = np.abs(np.sum(g_wn_up - g_wn_up_old))
        g_diff_dn = np.abs(np.sum(g_wn_dn - g_wn_dn_old))
        file.write(str(n_up)+"\t"+str(n_dn)+"\t"+str(g_diff_up)+"\t"+str(g_diff_dn)+"\n")
        
    file.close()
    return g_wn_up, g_wn_dn, sigma_wn_up, sigma_wn_dn