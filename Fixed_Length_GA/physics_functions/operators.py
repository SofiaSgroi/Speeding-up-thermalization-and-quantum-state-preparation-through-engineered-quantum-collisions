with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

import numpy as np
np.random.seed(seed_value)


def init_system_operators(N_levs=20, Omega_S=1., npdtype="complex64"):
        
    a = np.zeros([N_levs,N_levs], dtype=npdtype)
    for n_lev in range(1,N_levs):
        a[n_lev-1,n_lev] = np.sqrt(n_lev)
    a_dag = a.transpose().conjugate()

    H_sys = Omega_S*(a_dag@a + 0.5*np.eye(N_levs, dtype=npdtype))

    return a, a_dag, H_sys

def init_ancilla_operators(npdtype="complex64"):
    sx = np.array([[0,1],[1,0]], dtype=npdtype)
    sy = np.array([[0,-1j],[1j,0]], dtype=npdtype)
    sz = np.array([[1,0],[0,-1]], dtype=npdtype)
    Pauli = [sx, sy, sz]

    sp = np.array([[0,1],[0,0]], dtype=npdtype)
    sm = np.array([[0,0],[1,0]], dtype=npdtype)

    H0_A = 0.5*sz

    return Pauli, [sp, sm], H0_A




