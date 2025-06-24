with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

import numpy as np
np.random.seed(seed_value)

from scipy.linalg import expm

import physics_functions

class model_3levsAncillas:
        def __init__(self, rho0_tuple, N_levs=20, Omega_S=1., T=5., linear_interaction=True, omegaA_range_list = [10., 10.], grange_list = [1., 1., 1.], npdtype="complex64"):
                
                self.npdtype = npdtype
                self.N_levs = N_levs
                self.Omega_S = Omega_S
                self.T = T
                self.omegaA_range_list = omegaA_range_list
                self.grange_list = grange_list
                

                self.a, self.a_dag, self.H_sys = physics_functions.init_system_operators(N_levs, Omega_S, npdtype)
                self.GM_lambda, A_int, self.H0_A = physics_functions.init_ancilla_operators(3, npdtype)
                self.A_0to1, self.A_1to2, self.A_0to2, self.A_1to0, self.A_2to1, self.A_2to0 = A_int[0], A_int[1], A_int[2], A_int[3], A_int[4], A_int[5]

                if linear_interaction:
                        self.HI_list = [(np.kron(self.a_dag, self.A_1to0) + np.kron(self.a, self.A_0to1)), (np.kron(self.a_dag, self.A_2to1) + np.kron(self.a, self.A_1to2))]
                else:
                        self.HI_list = [(np.kron(self.a_dag, self.A_1to0) + np.kron(self.a, self.A_0to1)), (np.kron(self.a_dag, self.A_2to1) + np.kron(self.a, self.A_1to2)), (np.kron((self.a_dag@self.a_dag), self.A_2to0) + np.kron((self.a@self.a), self.A_0to2))]
                

                self.rhoS_0 = physics_functions.init_system_state(self.H_sys, self.a, self.a_dag, rho0_tuple)

                self.nparams_common = len(self.HI_list) + 2
                self.nparams_step = 8


        def collision(self, rhoS, rhoA, U_c): 
                rho = U_c@(np.kron(rhoS, rhoA))@(U_c.transpose().conjugate())
                return np.trace(rho.reshape(self.N_levs, 3, self.N_levs, 3), axis1=1, axis2=3)


        def get_rho(self, string):
                rtheta = np.pi*string[0]
                rphi = 2.*np.pi*string[1]
    
                rho_D11 = (np.sin(rtheta)*np.cos(rphi))**2
                rho_D22 = (np.sin(rtheta)*np.sin(rphi))**2
                rho_D33 = (np.cos(rtheta))**2
                rho_D = np.array([
                        [rho_D11, 0, 0],
                        [0, rho_D22, 0],
                        [0, 0, rho_D33]
                        ], dtype=self.npdtype)
    
                ualpha = np.pi*string[2]
                ubeta = 0.5*np.pi*string[3]
                ugamma = np.pi*string[4]
                utheta = 0.5*np.pi*string[5]
                ua = np.pi*string[6]
                ub = 0.5*np.pi*string[7]
    
                U_params = expm(1j*self.GM_lambda[3]*ualpha)@expm(1j*self.GM_lambda[2]*ubeta)@expm(1j*self.GM_lambda[3]*ugamma)@expm(1j*self.GM_lambda[5]*utheta)@expm(1j*self.GM_lambda[3]*ua)@expm(1j*self.GM_lambda[2]*ub)
    
                rhoA = U_params@rho_D@(U_params.T.conjugate())
                
                return rhoA
        
        def evolve(self, string):
                N_c = int((string.shape[0]-self.nparams_common)/self.nparams_step)
                t_c = float(self.T/N_c)
               
                omega_A11 = self.omegaA_range_list[0]*string[0]
                omega_A22 = omega_A11 + self.omegaA_range_list[1]*string[1]
                H_A = omega_A11*self.H0_A[0] + omega_A22*self.H0_A[1]
                H_I = np.zeros([3*self.N_levs, 3*self.N_levs], dtype=self.npdtype)
                for param_HI in range(len(self.HI_list)):
                        g_I = -self.grange_list[param_HI] + 2.*self.grange_list[param_HI]*string[param_HI+2]
                        H_I += g_I*self.HI_list[param_HI]
               
                H_tot = np.kron(self.H_sys, np.eye(3, dtype=self.npdtype)) + np.kron(np.eye(self.N_levs, dtype=self.npdtype), H_A) + H_I
                U_t = expm(-1j*H_tot*t_c)

                params_t = string[self.nparams_common:].reshape(N_c, self.nparams_step)
                rhoS_n = self.rhoS_0
                rhoS_list = [rhoS_n]
                for n_c in range(N_c):
                        rhoA = self.get_rho(params_t[n_c, :])
                        rhoS_n = self.collision(rhoS_n, rhoA, U_t)
                        rhoS_list += [rhoS_n]
                return rhoS_list
        
        def read_string(self, string):
                N_c = int((string.shape[0]-self.nparams_common)/self.nparams_step)
                t_c = float(self.T/N_c)

                omega_A11 = self.omegaA_range_list[0]*string[0]
                omega_A22 = omega_A11 + self.omegaA_range_list[1]*string[1]
                g_I_list = []
                for param_HI in range(len(self.HI_list)):
                        g_I_list += [-self.grange_list[param_HI] + 2.*self.grange_list[param_HI]*string[param_HI+2]]
                
                params_t = string[self.nparams_common:].reshape(N_c, self.nparams_step)
                rhoA_list = []
                for n_c in range(N_c):
                        rhoA_list += [self.get_rho(params_t[n_c, :])]
                
                return N_c, t_c, [omega_A11, omega_A22], g_I_list, rhoA_list
                        

     
                

                



