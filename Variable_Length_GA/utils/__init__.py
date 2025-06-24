with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

import numpy as np
np.random.seed(seed_value)

import os
import shutil

from scipy.linalg import expm, sqrtm, logm


import physics_functions

from .fitness import generate_ffunction
from .save_results import save_results