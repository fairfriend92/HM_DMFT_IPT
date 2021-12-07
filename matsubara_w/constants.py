import numpy as np

# Parameters
t = 0.5         # Hopping
D = 2 * t       # Half-bandwidth
N = 256         # Number of Matsubara frequencies
hyst = False    # If true loop for decreasing U   
do_pade = False # If true use Pade's continuation (not working!)

# Electron interaction
U_min = 0.1
dU = 0.1
U_max = 3.5
U_list = np.arange(U_min, U_max, dU) 
U_print = U_list   
if (hyst):
    U_list = np.append(U_list, U_print[::-1])
    U_print = np.append(U_print, U_print[::-1])

# Inverse of temperature 
beta_min = 4.
beta_max = 160.
dbeta = 8.
beta_list = np.arange(beta_min, beta_max, dbeta) #[48.]        
beta_print = beta_list     

# Real frequency
dw = 0.01                             
w = np.arange(-15, 15, dw)           

# Energy
de = 2.*t/256.                         
e = np.arange(-2*t, 2*t, de)         
dos_e = 2*np.sqrt(D**2 - e**2) / (np.pi * D**2) # Bethe lattice DOS