import numpy as np


min_c = 0
max_c = 40
data_points = 20

safe_c_step = 2

sweep_cs = np.sqrt(np.linspace(min_c ** 2, max_c ** 2, num=data_points))
while np.any(np.diff(sweep_cs) >= safe_c_step):
    c_steps = np.diff(sweep_cs)
    i = np.argmax(c_steps)
    max_step = c_steps[i]

    sweep_cs = np.insert(sweep_cs, i + 1, np.linspace(sweep_cs[i], sweep_cs[i + 1], 2+int(max_step // safe_c_step))[1:-1])