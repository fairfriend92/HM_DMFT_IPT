import numpy as np
import matplotlib.pylab as plt
import green_func as green_f
import print_func as print_f

def single_band_ipt_solver(u_int, g_0_wn_up, g_0_wn_dn, wn, tau, n_up, n_dn, loop):    
    beta = tau[1] + tau[-1]
    #beta = tau[-1]
    mu = u_int / 2
        
    # The Fourier transforms use as tail expansion of the atomic limit self-enegy
    # \Sigma(i\omega_n\rightarrow \infty) = \frac{U^2}{4(i\omega_n)}    
    g_0_tau_up = green_f.ift(wn, g_0_wn_up, tau, beta)
    g_0_tb_up = np.array([g_0_tau_up[len(tau)-1-t] for t in range(len(tau))])
    g_0_tau_dn = green_f.ift(wn, g_0_wn_dn, tau, beta)
    g_0_tb_dn = np.array([g_0_tau_dn[len(tau)-1-t] for t in range(len(tau))])   
    
    # IPT self-energy using G0 of quantum impurity
    sigma_tau_up = u_int**2 * g_0_tau_up * g_0_tb_dn * g_0_tau_dn
    sigma_tau_dn = u_int**2 * g_0_tau_dn * g_0_tb_up * g_0_tau_up
    sigma_wn_up = green_f.ft(wn, sigma_tau_up, tau, beta)
    sigma_wn_dn = green_f.ft(wn, sigma_tau_dn, tau, beta)
    
    '''
    # Not working for paramagnetic case
    n_0_up = 2/beta*np.sum(g_0_wn_up) + 0.5
    n_0_dn = 2/beta*np.sum(g_0_wn_dn) + 0.5s
    A_up = n_dn * (1 - n_dn) / (n_0_dn * (1 - n_0_dn))
    A_dn = n_up * (1 - n_up) / (n_0_up * (1 - n_0_up))
    B_up = (u_int * (1 - n_dn) - mu) / (u_int**2 * n_0_dn * (1 - n_0_dn))
    B_dn = (u_int * (1 - n_up) - mu) / (u_int**2 * n_0_up * (1 - n_0_up))
    sigma_wn_up = A_up * sigma_wn_up/(1 - B_up * sigma_wn_up)
    sigma_wn_dn = A_dn * sigma_wn_dn/(1 - B_dn * sigma_wn_dn)

    # Soumen's implementation  
    zeta_up = wn*1.j + mu - u_int*n_dn - sigma_wn_up
    zeta_dn = wn*1.j + mu - u_int*n_up - sigma_wn_dn
    g_e_up = zeta_dn/np.add.outer(-e**2, (zeta_dn*zeta_up))
    g_e_dn = zeta_up/np.add.outer(-e**2, (zeta_dn*zeta_up))
    dos_de = (dos_e * de).reshape(-1, 1)
    g_wn_up = (dos_de * g_e_up).sum(axis=0)
    g_wn_dn = (dos_de * g_e_dn).sum(axis=0)
    '''
                            
    # Dyson eq.
    g_wn_up = g_0_wn_up / (1.0 - sigma_wn_up * g_0_wn_up)
    g_wn_dn = g_0_wn_dn / (1.0 - sigma_wn_dn * g_0_wn_dn)  

    # Print G and self-energy
    print_f.generic(tau, g_0_tau_up, g_0_tau_dn, 
                    r'$\tau$', r'$G_0(\tau)$', 
                    "./figures/g_tau_not_converged/g_tau_loop="+str(loop)+".png")
    
    print_f.generic(tau, sigma_tau_up, sigma_tau_dn, 
                    r'$\tau$', r'$\Sigma_0(\tau)$', 
                    "./figures/sig_tau_not_converged/sig_tau_loop="+str(loop)+".png")

    return g_wn_up, g_wn_dn, sigma_wn_up, sigma_wn_dn
    
def loop(u_int, t, g_wn_up, g_wn_dn, wn, tau, mix=1, conv=1e-3):
    converged = False
    loops = 0
    iwn = 1j * wn
    beta = tau[1] + tau[-1]
    #beta = tau[-1]
    mu = u_int/2
    
    file = open("./data/dmft_loop_beta="+f'{beta:.3}'+"_U="+f'{u_int:.3}'+".txt", "w") 
    file.write("n_up\tn_dn\tg_diff_up\tg_diff_dn\n")
    
    while not converged:
        # Initial condition for the magnetization
        m = 0.0 if loops == 0 else 0.0
        
        # Backup old g
        g_wn_up_old = g_wn_up.copy()
        g_wn_dn_old = g_wn_dn.copy()
        
        # Occupation numbers
        n_up = 2/beta*np.sum(g_wn_up_old.real) + 0.5
        n_dn = 2/beta*np.sum(g_wn_dn_old.real) + 0.5  
        
        # Non-interacting GF of quantum impurity  
        g_0_wn_up = 1. / (iwn + m - t**2 * g_wn_up_old + mu - u_int*n_dn)
        g_0_wn_dn = 1. / (iwn - m - t**2 * g_wn_dn_old + mu - u_int*n_up)
        
        # Impurity solver
        g_wn_up, g_wn_dn, sigma_wn_up, sigma_wn_dn = \
            single_band_ipt_solver(u_int, g_0_wn_up, g_0_wn_dn, wn, tau, n_up, n_dn, loops)  
        
        # Clean for Half-fill
        g_wn_up.real = g_wn_dn.real = 0.
        
        # Check convergence
        converged = np.allclose(g_wn_up_old, g_wn_up, conv) and np.allclose(g_wn_dn_old, g_wn_dn, conv)
        loops += 1
        if loops > 50:
            converged = True
        g_wn_up = mix * g_wn_up + (1 - mix) * g_wn_up_old
        g_wn_dn = mix * g_wn_dn + (1 - mix) * g_wn_dn_old
        
        # Write datafile 
        g_diff_up = np.abs(np.sum(g_wn_up - g_wn_up_old))
        g_diff_dn = np.abs(np.sum(g_wn_dn - g_wn_dn_old))
        file.write(str(n_up)+"\t"+str(n_dn)+"\t"+str(g_diff_up)+"\t"+str(g_diff_dn)+"\n")
        
    file.close()
    return g_wn_up, g_wn_dn, sigma_wn_up, sigma_wn_dn