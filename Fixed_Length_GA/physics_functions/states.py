with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

import numpy as np
np.random.seed(seed_value)

from scipy.linalg import expm


def thermal(H_sys, beta):
        thstate = expm(-beta*H_sys)
        return thstate/np.trace(thstate)

def number(H_sys, n_exc):
        ndims = H_sys.shape[0]
        npdtype = H_sys.dtype
        nstate = np.zeros([ndims, ndims], dtype=npdtype)
        nstate[int(n_exc), int(n_exc)] = 1.
        return nstate

def Displace(rho_in, alpha, a, a_dag):
    D_alpha = expm(alpha*a_dag-alpha.conjugate()*a)
    rho_out = D_alpha@rho_in@(D_alpha.conjugate().transpose())
    rho_out = rho_out/np.trace(rho_out)
    return rho_out

def Squeeze(rho_in, zeta, a, a_dag):
    Sqz = expm(0.5*(zeta.conjugate()*a@a - zeta*a_dag@a_dag))
    rho_out = Sqz@rho_in@(Sqz.conjugate().transpose())
    rho_out = rho_out/np.trace(rho_out)
    return  rho_out

def init_system_state(H_sys, a, a_dag, starting_state, operations_list=[]):
        #(state_type, parameter), [(operation, parameter)]
        if starting_state[0]=="thermal":
                rhoS_X = thermal(H_sys, starting_state[1])
        elif starting_state[0]=="number":
                rhoS_X = number(H_sys, starting_state[1])
        else:
                print("invalid starting state")

        for operation in operations_list:
                if operation[0]=="D":
                        rhoS_X = Displace(rhoS_X, operation[1], a, a_dag)
                elif operation[0]=="S":
                        rhoS_X = Squeeze(rhoS_X, operation[1], a, a_dag)
                else:
                        print(operation[0], " is not a valid operation:")
                        
        return rhoS_X