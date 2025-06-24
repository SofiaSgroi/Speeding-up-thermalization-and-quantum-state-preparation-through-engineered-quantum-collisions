with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

from hyperparameters.hyperparameters_physics import objective_precision

import numpy as np
np.random.seed(seed_value)

from scipy.linalg import sqrtm, logm

def trace_distance(rho_1, rho_2):
    rhodiff = (rho_1-rho_2).round(objective_precision)
    return 0.5*((np.absolute(np.linalg.eigvals(rhodiff))).sum()).round(objective_precision)



