with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

import numpy as np
np.random.seed(seed_value)

import os
import shutil


def save_results(folder_name, solution, training_perf):
        os.mkdir(folder_name)

        os.mkdir("info")
        shutil.copy("hyperparameters/hyperparameters_GA.py", "info")
        shutil.copy("hyperparameters/hyperparameters_physics.py", "info")
        shutil.copy("hyperparameters/seedfile.txt", "info")
        shutil.copy("config.py", "info")
        shutil.move("info", folder_name)

        np.savetxt((folder_name + "/solution"), solution)
        np.savetxt((folder_name + "/history"), training_perf)

        shutil.move(folder_name, "results")   
