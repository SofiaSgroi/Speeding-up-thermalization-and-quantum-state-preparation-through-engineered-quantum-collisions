folder_name ="twoAlevs10Slevs_linear_n1T2_v2"
nsteps = 500

ancilla_levs = 2

Omega_S = 1.
rho0_tuple = ("number", 0)
T = 2.

linear_interaction = True

objective ="minimize trace distance" #"maximize non gaussianity"
target = [("number", 1)] #[("number", 0), ("S", 1.)]


