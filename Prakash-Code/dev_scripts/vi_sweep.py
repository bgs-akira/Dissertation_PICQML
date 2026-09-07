import numpy as np
import matplotlib.pyplot as plt

from prakash.mesh import Mesh
import prakash.config as config
from prakash.utils import CaliUtils, DFUtils

pMesh = Mesh()

num_cs = 10
sweep_cs = np.linspace(1, 40, num=num_cs)
sweep_v = 20

read_nums = 10
sleep_t = 0.05

plt.figure('VI plot for 100 phaseshifters')

total_results = np.zeros((100, num_cs, 5))
total_stds = np.zeros((100, num_cs, 5))
i_iter = 0

for i in range(10):
    for j in range(10):
        phaseshifter = (i,j)

        c_results = pMesh.sweep_current(phaseshifter=phaseshifter, sweep_cs=sweep_cs, sweep_v=sweep_v, read_nums=read_nums, sleep_t=sleep_t)

        plotting_dir = rf'..\Results\test_calibration\phaseshifter{phaseshifter}'
        # CaliUtils.plot_phaseshifter_cali(phaseshifter=phaseshifter, c_results=c_results, results_dir=plotting_dir)
        np.save(DFUtils.create_filename(plotting_dir + r'\c_results.npy'), c_results)

        # each row is a different sweep_c, storing [sweep_c, mean_c, mean_v, mean_electric_power, mean_resistance]
        mean_results = np.mean(c_results, axis=1)
        std_results = np.std(c_results, axis=1)

        total_results[i_iter, :, :] = mean_results
        total_stds[i_iter, :, :] = std_results

        plt.errorbar(mean_results[:, 1], mean_results[:, 2], xerr=std_results[:, 1], yerr=std_results[:, 2],
                     fmt='-', lw=1,)

        i_iter += 1

np.save(rf'..\Results\test_calibration\all_mean_results_{config.time_stamp}.npy', total_results)
np.save(rf'..\Results\test_calibration\all_std_results_{config.time_stamp}.npy', total_stds)

plt.xlabel('Measured current (mA)')
plt.ylabel('Measured voltage (V)')

plt.show()
plt.savefig(rf'..\Results\test_calibration\vi_plot_100_phaseshifters.png')