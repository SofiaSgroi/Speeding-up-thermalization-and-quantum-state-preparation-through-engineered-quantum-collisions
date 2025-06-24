with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

import numpy as np
np.random.seed(seed_value)

from scipy.linalg import expm, sqrtm, logm


import physics_functions

def generate_ffunction(model, objective="minimize trace distance", target=[]):
    if objective=="minimize trace distance":
        target_starting_state, target_operations_list= target[0], target[1:]
        target_state = physics_functions.init_system_state(model.H_sys, model.a, model.a_dag, target_starting_state, target_operations_list)
        def fitness_function(string):
            rhoS_f = model.evolve(string)[-1]
            trD_best = physics_functions.trace_distance(rhoS_f, target_state)
            return -trD_best
    elif objective=="maximize non gaussianity":
         def fitness_function(string):
            rhoS_f = model.evolve(string)[-1]
            ng_best = physics_functions.nongaussianity(rhoS_f, physics_functions.get_CovarianceMatrix(rhoS_f, model.a, model.a_dag))
            return ng_best
    return fitness_function