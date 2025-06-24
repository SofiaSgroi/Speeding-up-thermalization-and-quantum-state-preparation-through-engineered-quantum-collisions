with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

import numpy as np
np.random.seed(seed_value)

import os
import shutil

from tqdm import tqdm

from scipy.linalg import expm, sqrtm, logm

import GA
import physics_functions
import physics_classes
import utils

from hyperparameters.hyperparameters_GA import *
from hyperparameters.hyperparameters_physics import *
from config import *

if thermal_ancillas:
        model = physics_classes.model_thermalAncillas(rho0_tuple, N_levs, Omega_S, t_c, g, beta_range, npdtype)
else:
        model = physics_classes.model_genericAncillas(rho0_tuple, N_levs, Omega_S, t_c, g, npdtype)

fitnessf = utils.generate_ffunction(model, target)
max_length = N_c
elementcells = model.nparams_step
common = model.nparams_common

genes = GA.GeneticAlgorithm(fitnessf, pop_size, max_length, elementcells, mutation_factor, K, nbests)
genes.init_pop()
genes.optimize(nsteps)
solution, training_perf =  genes.results()
print(training_perf[-1])

utils.save_results(folder_name, solution, np.array(training_perf))
