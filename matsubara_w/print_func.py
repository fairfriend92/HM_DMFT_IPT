import matplotlib.pylab as plt

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

# Print density of states
def dos(beta_print, dos_U, U_print, hyst):
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

# Print the Green functions
def green_func(beta_print, tau_U, \
               g_wn_U_up, g_wn_U_dn, g_tau_U_up, g_tau_U_dn, 
               U_print, hyst, wn):
    for i in range(len(beta_print)):
        print("Printing Green functions")
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
    for i in range(len(beta_print)):
        print("Printing zero freqeuncy Matsubara g")
        beta = beta_print[i]
        g_wn = g_wn_U_up[i]
        Gw0 = []
        for g in g_wn:
            Gw0.append(g[0].imag)        
        plt.figure()
        plt.xlabel(r'$U$')
        plt.ylabel(r'$g(\omega_0)$')
        plt.plot(U_print, Gw0)
        plt.savefig("./figures/g_w0/g_w0_beta="+f'{beta:.3}'+".png")
        plt.close()

# Print electron occupation
def n(beta_print, n_U, U_print):
    plt.figure()
    for i in range(len(beta_print)):
        print("Printing e concentration")
        n = n_U[i]
        beta = beta_print[i]    
        plt.xlabel('U')
        plt.ylabel('n')
        plt.plot(U_print, n, label='beta='+f'{beta:.3}')
        plt.legend()
    plt.savefig("./figures/n.png")

# Print double occupancy
def d(beta_print, d_U, U_print):
    plt.figure()
    for i in range(len(beta_print)):
        print("Printing double occupancy")
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
    for i in range(len(beta_print)):
        print("Printing kinetic energy")
        e_kin = ekin_U[i]
        beta = beta_print[i]
        plt.xlabel('U')
        plt.ylabel(r'$E_K$')
        plt.xlim(2, 3.5)
        plt.ylim(-0.5, 0)
        plt.plot(U_print, e_kin, label='beta='+f'{beta:.3}')
        plt.legend()
    plt.savefig("./figures/e_kin.png")

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
    if np.abs(g_wn[T][U][0].imag) < val:
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
                if get_phase(i, j, 0.15, g_wn_U_up) != \
                   get_phase(i, j+1, 0.15, g_wn_U_up):
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
    plt.savefig("./figures/phase_diag.png")