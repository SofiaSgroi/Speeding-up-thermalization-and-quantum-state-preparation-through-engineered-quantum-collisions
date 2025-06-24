with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

import numpy as np
np.random.seed(seed_value)

from scipy.linalg import expm

from .model_3levsAncillas import model_3levsAncillas
from .model_qubitAncillas import model_qubitAncillas