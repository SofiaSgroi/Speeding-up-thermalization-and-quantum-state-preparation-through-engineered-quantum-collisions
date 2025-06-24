folder_name ="threeAlevs20Slevs_linear_squeezed05T5"
nsteps = 2500

ancilla_levs = 3

Omega_S = 1.
rho0_tuple = ("number", 0)
T = 5.

linear_interaction = True

objective = "minimize trace distance" #"maximize non gaussianity", "minimize trace distance" 
target = [("number", 0), ("S", 0.5)] #[("number", 0), ("S", 1.)]


