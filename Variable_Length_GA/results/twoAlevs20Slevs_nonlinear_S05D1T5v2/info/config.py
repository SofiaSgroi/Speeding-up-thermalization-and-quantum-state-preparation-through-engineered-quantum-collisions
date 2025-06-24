folder_name ="twoAlevs20Slevs_nonlinear_S05D1T5v2"
nsteps = 3000

ancilla_levs = 2

Omega_S = 1.
rho0_tuple = ("number", 0)
T = 5.

linear_interaction = False

objective = "minimize trace distance" #"maximize non gaussianity", "minimize trace distance" 
target = [("number", 0), ("S", 0.5), ("D", 1.)] #[("number", 0), ("S", 1.)]


