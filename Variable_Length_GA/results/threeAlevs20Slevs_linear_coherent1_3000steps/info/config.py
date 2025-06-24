folder_name ="threeAlevs20Slevs_linear_coherent1_3000steps"
nsteps = 3000

ancilla_levs = 3

Omega_S = 1.
rho0_tuple = ("number", 0)
T = 5.

linear_interaction = True

objective ="minimize trace distance" #"maximize non gaussianity"
target = [("number", 0), ("D", 1.)] #[("number", 0), ("S", 1.)]


