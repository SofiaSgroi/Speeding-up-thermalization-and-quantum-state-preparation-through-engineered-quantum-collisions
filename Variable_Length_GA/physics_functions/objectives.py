with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

from hyperparameters.hyperparameters_physics import objective_precision

import numpy as np
np.random.seed(seed_value)

from scipy.linalg import sqrtm, logm

def trace_distance(rho_1, rho_2):
    rhodiff = (rho_1-rho_2).round(objective_precision)
    return 0.5*((np.absolute(np.linalg.eigvals(rhodiff))).sum()).round(objective_precision)

def nongaussianity(rho_sys, covM):
      xCM_err = 10**(-objective_precision)
      log_err = (10**(-objective_precision))*np.eye(rho_sys.shape[0], dtype=rho_sys.dtype)

      DetCovM = covM[0,0]*covM[1,1] - covM[0, 1]*covM[1, 0]
      xCM = np.sqrt(DetCovM).round(objective_precision) + xCM_err
      h_CM = (xCM + 0.5)*np.log(xCM + 0.5) - (xCM - 0.5)*np.log(xCM - 0.5)
      log_rho = logm(rho_sys.round(objective_precision) + log_err)
      NG_relentropy = h_CM + np.trace(rho_sys@log_rho).real
      return NG_relentropy.round(objective_precision)


