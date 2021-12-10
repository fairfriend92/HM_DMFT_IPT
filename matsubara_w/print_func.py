import matplotlib.pylab as plt
import numpy as np
from constants import *

# Print any 2 complex functions  
def generic(x, y_up, y_dn, x_label, y_label, path):
    plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y_up.imag, s=1, label=r'$\sigma=\uparrow$ Im')
    plt.scatter(x, y_up.real, s=1, label=r'$\sigma=\uparrow$ Re')
    plt.scatter(x, y_dn.imag, s=1, label=r'$\sigma=\downarrow$ Im')
    plt.scatter(x, y_dn.real, s=1, label=r'$\sigma=\downarrow$ Re')
    plt.legend()
    plt.savefig(path)
    plt.close()
    
def not_converged(wn, tau, g_0_wn_up, g_0_wn_dn, g_0_tau_up, g_0_tau_dn,
                 sigma_wn_up, sigma_wn_dn, 
                 sigma_tau_up, sigma_tau_dn, 
                 g_wn_up, g_wn_dn, loop, U):
    # Print g_0_wn
    generic(wn, g_0_wn_up, g_0_wn_dn, 
            r'$\omega_n$', r'$G_0(\omega_n)$', 
            "./figures/not_converged/g_0_wn_U="+f'{U:.3}'+"_loop="+str(loop)+".pdf")  
    
    # Write g_0_wn
    file = open("./data/not_converged/g_0_wn_U="+f'{U:.3}'+"_loop="+str(loop)+".txt", "w") 
    file.write("wn\tg_0_wn_up\tg_0_wn_dn\n")
    for w, g_up, g_dn in zip(wn, g_0_wn_up, g_0_wn_dn):
        file.write(str(w) + "\t" + str(g_up) + "\t" + str(g_dn) + "\n")
    file.close()
                    
    # Print g_0_tau                
    generic(tau, g_0_tau_up, g_0_tau_dn, 
            r'$\tau$', r'$G_0(\tau)$', 
            "./figures/not_converged/g_0_tau_U="+f'{U:.3}'+"_loop="+str(loop)+".pdf") 
    
    # Write g_0_tau     
    file = open("./data/not_converged/g_0_tau_U="+f'{U:.3}'+"_loop="+str(loop)+".txt", "w") 
    file.write("tau\tg_0_tau_up\tg_0_tau_up\n")
    for t, g_up, g_dn in zip(tau, g_0_tau_up, g_0_tau_dn):
        file.write(str(t) + "\t" + str(g_up) + "\t" + str(g_dn) + "\n")
    file.close()
    
    # Print sigma_wn 
    generic(wn, sigma_wn_up, sigma_wn_dn, 
            r'$\omega_n$', r'$\Sigma(\omega_n)$', 
            "./figures/not_converged/sig_wn_U="+f'{U:.3}'+"_loop="+str(loop)+".pdf")
            
    # Write sigma_wn
    file = open("./data/not_converged/sig_wn_U="+f'{U:.3}'+"_loop="+str(loop)+".txt", "w") 
    file.write("wn\tsig_0_wn_up\tsig_0_wn_dn\n")
    for w, sig_up, sig_dn in zip(wn, sigma_wn_up, sigma_wn_dn):
        file.write(str(w) + "\t" + str(sig_up) + "\t" + str(sig_dn) + "\n")
    file.close()
    
    # Print sigma_tau
    generic(tau, sigma_tau_up, sigma_tau_dn, 
            r'$\tau$', r'$\Sigma(\tau)$', 
            "./figures/not_converged/sig_tau_U="+f'{U:.3}'+"_loop="+str(loop)+".pdf")
            
    # Write sigma_tau
    file = open("./data/not_converged/sig_tau_U="+f'{U:.3}'+"_loop="+str(loop)+".txt", "w") 
    file.write("wn\tsig_0_tau_up\tsig_0_tau_dn\n")
    for w, sig_up, sig_dn in zip(wn, sigma_tau_up, sigma_tau_dn):
        file.write(str(w) + "\t" + str(sig_up) + "\t" + str(sig_dn) + "\n")
    file.close()
    
    # Print g_wn
    generic(wn, g_wn_up, g_wn_dn, 
            r'$\omega_n$', r'$G(\omega_n)$', 
            "./figures/not_converged/g_wn_U="+f'{U:.3}'+"_loop="+str(loop)+".pdf")
            
    # Write g_wn
    file = open("./data/not_converged/g_wn_U="+f'{U:.3}'+"_loop="+str(loop)+".txt", "w") 
    file.write("wn\tg_wn_up\tg_wn_dn\n")
    for w, g_up, g_dn in zip(wn, g_wn_up, g_wn_dn):
        file.write(str(w) + "\t" + str(g_up) + "\t" + str(g_dn) + "\n")
    file.close()

# Print density of states
def dos(beta_print, w, dos_U, U_print, hyst):
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
        plt.savefig("./figures/dos_beta="+f'{beta:.3}'+".pdf")
        plt.close()

# Print the Green functions
def green_func(beta_print, tau_U, \
               g_wn_U_up, g_wn_U_dn, g_tau_U_up, g_tau_U_dn, 
               U_print, hyst, wn):
    print("Printing Green functions")
    for i in range(len(beta_print)):
        beta = beta_print[i]
        tau = tau_U[i]
        g_wn_up = g_wn_U_up[i]
        g_wn_dn = g_wn_U_dn[i]
        g_tau_up = g_tau_U_up[i]
        g_tau_dn = g_tau_U_dn[i]
        for j in range(len(U_print)):
            U = U_print[j]
            
            if hyst:
                branch = "_up" if  j < len(U_print)/2 else "_dn"
            else:
                branch = ""
                               
            # Matsubara Green function
            plt.figure()
            plt.xlabel(r'$\omega_n$')
            plt.ylabel(r'$g(\omega_n)$')
            plt.scatter(wn, g_wn_up[j].imag, s=1,  label=r'$\sigma=\uparrow$ Im')
            plt.scatter(wn, g_wn_up[j].real, s=1,  label=r'$\sigma=\uparrow$ Re')
            plt.scatter(wn, g_wn_dn[j].imag, s=1,  label=r'$\sigma=\downarrow$ Im')
            plt.scatter(wn, g_wn_dn[j].real, s=1,  label=r'$\sigma=\downarrow$ Re')
            plt.legend()
            plt.title(r"$\beta$ = "+f'{beta:.3}'+" U = "+f'{U:.3}'+branch)
            plt.savefig("./figures/g_wn/g_wn_beta="+f'{beta:.3}'+"_U="+f'{U:.3}'+branch+".pdf")
            plt.close()
            
            file = open("./data/g_wn_beta="+f'{beta:.3}'+"_U="+f'{U:.3}'+".txt", "w") 
            file.write("wn\tg_wn_up\tg_wn_dn\n")
            for w, g_up, g_dn in zip(wn, g_wn_up[j], g_wn_dn[j]):
                file.write(str(w) + "\t" + str(g_up) + "\t" + str(g_dn) + "\n")
            file.close()
            
            # Imaginary time Green function
            plt.figure()
            plt.xlabel(r'$\tau$')
            plt.ylabel(r'$g(\tau)$')
            plt.scatter(tau, g_tau_up[j].imag, s=1, label=r'$\sigma=\uparrow$ Im')
            plt.scatter(tau, g_tau_up[j].real, s=1, label=r'$\sigma=\uparrow$ Re')
            plt.scatter(tau, g_tau_dn[j].imag, s=1, label=r'$\sigma=\downarrow$ Im')
            plt.scatter(tau, g_tau_dn[j].real, s=1, label=r'$\sigma=\downarrow$ Re')
            plt.legend()
            plt.title(r"$\beta$ = "+f'{beta:.3}'+" U = "+f'{U:.3}'+branch)
            plt.savefig("./figures/g_tau/g_tau_beta="+f'{beta:.3}'+"_U="+f'{U:.3}'+branch+".pdf")
            plt.close()

# Print zero-freq Matsubara Green function
def gf_iw0(beta_print, g_wn_U_up, U_print):
    print("Printing zero freqeuncy Matsubara g")
    for i in range(len(beta_print)):        
        beta = beta_print[i]
        g_wn = g_wn_U_up[i]
        Gw0 = []
        for g in g_wn:
            Gw0.append(g[int(N/2)].imag)        
        plt.figure()
        plt.xlabel(r'$U$')
        plt.ylabel(r'$g(\omega_0)$')
        plt.plot(U_print, Gw0)
        plt.savefig("./figures/g_w0/g_w0_beta="+f'{beta:.3}'+".png")
        plt.close()

# Print electron occupation
def n(beta_print, n_U, U_print):
    plt.figure()
    print("Printing e concentration")
    for i in range(len(beta_print)):        
        n = n_U[i]
        beta = beta_print[i]    
        plt.xlabel('U')
        plt.ylabel('n')
        plt.plot(U_print, n, label='beta='+f'{beta:.3}')
        plt.ylim(0.0, 1.0)
        plt.legend()
    plt.savefig("./figures/n.png")

# Print double occupancy
def d(beta_print, d_U, U_print):
    plt.figure()
    print("Printing double occupancy")
    for i in range(len(beta_print)):        
        d = d_U[i]
        beta = beta_print[i]    
        plt.xlabel('U')
        plt.ylabel('d')
        plt.plot(U_print, d, label='beta='+f'{beta:.3}')
        plt.legend()
    plt.savefig("./figures/d.png")

# Print kinetic energy
def e_kin(beta_print, ekin_U, U_print):
    plt.figure()
    print("Printing kinetic energy")
    for i in range(len(beta_print)):        
        e_kin = ekin_U[i]
        beta = beta_print[i]
        plt.xlabel('U')
        plt.ylabel(r'$E_K$')
        plt.xlim(2, 3.5)
        plt.ylim(-0.5, 0)
        plt.plot(U_print, e_kin, label='beta='+f'{beta:.3}')
        plt.legend()
    plt.savefig("./figures/e_kin.pdf")

# Print quasi-particle weight
def Z(beta_print, Z_U, U_print):
    plt.figure()
    for i in range(len(beta_print)):
        Z = Z_U[i]
        beta = beta_print[i]
        plt.xlabel('U')
        plt.ylabel('Z')
        plt.plot(U_print, Z, label='beta='+f'{beta:.3}')
        plt.legend()
    plt.savefig("./figures/Z.png")

def get_phase(U, T, val, g_wn_U_up):
    # g for different T, U values
    g_wn = np.flipud(g_wn_U_up) # Increasing temp order
    if np.abs(g_wn[T][U][int(N)].imag) < val:
        return -1             # Metallic phase
    else:
        return 1              # Insulating phase

# Print phase diagram
def phase(beta_list, U_print, g_wn_U_up):
    if (len(beta_list) > 1):
        plt.figure()
        plt.xlabel('U')
        plt.ylabel('T')
        T_list = [1/beta for beta in beta_list]    # Convert from beta to temp
        T_list = np.flipud(T_list)                 # Sort in increasing order
        T_trans = [0.]                             # Transition temperature
           
        for i in range(len(U_print)):
            for j in range(len(T_list)-1):
                if get_phase(i, j, 1.0, g_wn_U_up) != \
                   get_phase(i, j+1, 1.0, g_wn_U_up):
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
        plt.plot(U_mask, T_mask, marker='.')
        plt.savefig("./figures/phase_diag.pdf")