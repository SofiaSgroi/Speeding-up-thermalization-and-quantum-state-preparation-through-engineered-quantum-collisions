with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

import numpy as np
np.random.seed(seed_value)

from scipy.linalg import expm

import physics_functions


class model_genericAncillas:
        def __init__(self, rho0_tuple, N_levs=20, Omega_S=1., t_c = 0.1, g = 1., npdtype="complex64"):

                self.npdtype = npdtype
                self.N_levs = N_levs
                self.Omega_S = Omega_S
                self.t_c = t_c
                self.g = g
                self.nparams_common = 0
                self.nparams_step = 3
                

                self.a, self.a_dag, self.H_sys = physics_functions.init_system_operators(N_levs, Omega_S, npdtype)
                self.Pauli, A_int, self.H0_A = physics_functions.init_ancilla_operators(npdtype)
                self.sp, self.sm = A_int[0], A_int[1]

                self.H_I = self.g*(np.kron(self.a_dag, self.sm) + np.kron(self.a, self.sp))
                
                self.H_A = self.Omega_S*self.H0_A
               
                self.H_tot = np.kron(self.H_sys, np.eye(2, dtype=self.npdtype)) + np.kron(np.eye(self.N_levs, dtype=self.npdtype), self.H_A) + self.H_I
                self.U_t = expm(-1j*self.H_tot*self.t_c)

                self.rhoS_0 = physics_functions.init_system_state(self.H_sys, self.a, self.a_dag, rho0_tuple)



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
                N_c = int(string.shape[0]/self.nparams_step)
                params_t = string.reshape(N_c, self.nparams_step)

                rhoS_n = self.rhoS_0
                rhoS_list = [rhoS_n]
                for n_c in range(N_c):
                        rhoA = self.get_rho(params_t[n_c, :])
                        rhoS_n = self.collision(rhoS_n, rhoA, self.U_t)
                        rhoS_list += [rhoS_n]
                return rhoS_list
                        

     
                

                



