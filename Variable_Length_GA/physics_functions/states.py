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

def get_CovarianceMatrix(rho_sys, a, a_dag):
        X_op = (a + a_dag)/np.sqrt(2.)
        P_op = 1j*(a_dag - a)/np.sqrt(2.)
       
        X2_op = X_op@X_op
        P2_op = P_op@P_op
        XP_op = X_op@P_op
        PX_op = P_op@X_op

        X_avg = np.trace(rho_sys@X_op).real
        P_avg = np.trace(rho_sys@P_op).real
        X2_avg = np.trace(rho_sys@X2_op).real
        P2_avg = np.trace(rho_sys@P2_op).real
        XP_avg = np.trace(rho_sys@XP_op).real
        PX_avg = np.trace(rho_sys@PX_op).real

        sigma_XX = X2_avg - X_avg*X_avg
        sigma_XP = 0.5*(XP_avg + PX_avg) - X_avg*P_avg
        sigma_PX = sigma_XP
        sigma_PP = P2_avg - P_avg*P_avg

        CovM = np.array([[sigma_XX, sigma_XP],[sigma_PX, sigma_PP]])
        return CovM