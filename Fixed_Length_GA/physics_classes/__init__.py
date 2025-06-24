with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

import numpy as np
np.random.seed(seed_value)

from scipy.linalg import expm

from .model_thermalAncillas import model_thermalAncillas
from .model_genericAncillas import model_genericAncillas