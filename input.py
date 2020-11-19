import numpy as np



R_c = 6.5
R_s = 0

def cutoff_function(R_ij,R_c):
    if R_ij <= R_c:
        return 0.5 * (np.cos((np.pi * R_ij)/R_c)+1)
    else:
        return 0

print(cutoff_function(1,5))

def radial_distribution():
    g_r = np.exp(-1 * eeta * (R_ij-R_s)**2) * cutoff_function(R_ij,R_c)
    #G = sum of g s
    pass


def angular_distribution():
    g_a = ((1 + lamda*np.cos(theeta))**zeta) * np.exp(-1*eeta*(R_ij**2+R_ik**2+R_jk**2)) * cutoff_function(R_ij,R_c)* cutoff_function(R_ik,R_c)*cutoff_function(R_jk,R_c)
    pass