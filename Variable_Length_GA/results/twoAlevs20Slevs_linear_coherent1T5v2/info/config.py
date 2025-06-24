folder_name ="twoAlevs20Slevs_linear_coherent1T5v2"
nsteps = 2000

ancilla_levs = 2

Omega_S = 1.
rho0_tuple = ("number", 0)
T = 5.

linear_interaction = True

objective ="minimize trace distance" #"maximize non gaussianity"
target = [("number", 0), ("D", 1.)] #[("number", 0), ("S", 1.)]


