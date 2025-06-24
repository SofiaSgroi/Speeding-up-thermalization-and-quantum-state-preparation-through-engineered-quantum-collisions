folder_name ="twoAlevs30Slevs_nonlinear_squeezed1T5"
nsteps = 2000

ancilla_levs = 2

Omega_S = 1.
rho0_tuple = ("number", 0)
T = 5.

linear_interaction = False

objective ="minimize trace distance" #"maximize non gaussianity"
target = [("number", 0), ("S", 1.)] #[("number", 0), ("S", 1.)]


