folder_name ="twoAlevs30Slevs_nonlinear_squeezed1T10_strongercoupling"
nsteps = 2500

ancilla_levs = 2

Omega_S = 1.
rho0_tuple = ("number", 0)
T = 10.

linear_interaction = False

objective ="minimize trace distance" #"maximize non gaussianity"
target = [("number", 0), ("S", 1.)] #[("number", 0), ("S", 1.)]


