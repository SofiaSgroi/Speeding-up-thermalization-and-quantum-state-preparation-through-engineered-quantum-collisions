folder_name ="twoAlevs20Slevs_linear_squeezed05T5"
nsteps = 2000

ancilla_levs = 2

Omega_S = 1.
rho0_tuple = ("number", 0)
T = 5.

linear_interaction = True

objective ="minimize trace distance" #"maximize non gaussianity"
target = [("number", 0), ("S", 0.5)] #[("number", 0), ("S", 1.)]


