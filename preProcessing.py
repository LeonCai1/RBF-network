import numpy as np
import random
# create a array with 75 row and 2 col
data = np.random.uniform(0, 1, (75, 1))
desired = 0.5 + 0.4 * np.sin(2 * np.pi * data) + np.random.uniform(-0.1, 0.1, (75, 1))

