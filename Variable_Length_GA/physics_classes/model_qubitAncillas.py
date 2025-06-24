with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

import numpy as np
np.random.seed(seed_value)

from scipy.linalg import expm

import physics_functions


class model_qubitAncillas:
        def __init__(self, rho0_tuple, N_levs=20, Omega_S=1., T=5., linear_interaction=True, omegaA_range_list = [10.], grange_list = [1., 1.], npdtype="complex64"):
                
                self.npdtype = npdtype
                self.N_levs = N_levs
                self.Omega_S = Omega_S
                self.T = T
                self.omegaA_range = omegaA_range_list[0]
                self.grange_list = grange_list
                

                self.a, self.a_dag, self.H_sys = physics_functions.init_system_operators(N_levs, Omega_S, npdtype)
                self.Pauli, A_int, self.H0_A = physics_functions.init_ancilla_operators(2, npdtype)
                self.sp, self.sm = A_int[0], A_int[1]

                if linear_interaction:
                        self.HI_list = [(np.kron(self.a_dag, self.sm) + np.kron(self.a, self.sp))]
                else:
                        self.HI_list = [(np.kron(self.a_dag, self.sm) + np.kron(self.a, self.sp)), (np.kron(self.a_dag@self.a_dag, self.sm) + np.kron(self.a@self.a, self.sp))]
                

                self.rhoS_0 = physics_functions.init_system_state(self.H_sys, self.a, self.a_dag, rho0_tuple)

                self.nparams_common = len(self.HI_list) + 1
                self.nparams_step = 3


        def collision(self, rhoS, rhoA, U_c): 
                rho = U_c@(np.kron(rhoS, rhoA))@(U_c.transpose().conjugate())
                return np.trace(rho.reshape(self.N_levs, 2, self.N_levs, 2), axis1=1, axis2=3)


        def get_rho(self, string):
                r, theta, phi = string[0], string[1], string[2]

                vx = r*np.sin(np.pi*theta)*np.cos(2.*np.pi*phi)
                vy = r*np.sin(np.pi*theta)*np.sin(2.*np.pi*phi)
                vz = r*np.cos(np.pi*theta) 
        
                rhoA = 0.5*(np.eye(2, dtype=self.npdtype) + vx*self.Pauli[0] + vy*self.Pauli[1] + vz*self.Pauli[2])
                return rhoA
        
        def evolve(self, string):
                N_c = int((string.shape[0]-self.nparams_common)/self.nparams_step)
                t_c = float(self.T/N_c)
               
                omega_A = self.omegaA_range*string[0]
                H_A = omega_A*self.H0_A
                H_I = np.zeros([2*self.N_levs, 2*self.N_levs], dtype=self.npdtype)
                for param_HI in range(len(self.HI_list)):
                        g_I = -self.grange_list[param_HI] + 2.*self.grange_list[param_HI]*string[param_HI+1]
                        H_I += g_I*self.HI_list[param_HI]
               
                H_tot = np.kron(self.H_sys, np.eye(2, dtype=self.npdtype)) + np.kron(np.eye(self.N_levs, dtype=self.npdtype), H_A) + H_I
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
                
                omega_A = self.omegaA_range*string[0]
                g_I_list = []
                for param_HI in range(len(self.HI_list)):
                        g_I_list += [-self.grange_list[param_HI] + 2.*self.grange_list[param_HI]*string[param_HI+1]]
                
                params_t = string[self.nparams_common:].reshape(N_c, self.nparams_step)
                rhoA_list = []
                for n_c in range(N_c):
                        rhoA_list += [self.get_rho(params_t[n_c, :])]
                
                return N_c, t_c, omega_A, g_I_list, rhoA_list
                        

     
                

                



