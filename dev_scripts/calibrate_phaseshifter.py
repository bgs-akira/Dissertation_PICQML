import numpy as np
import matplotlib.pyplot as plt
import logging

from prakash.mesh import Mesh
import prakash.config as config
from dev_scripts.devices import powerSupply, opticalPowerMeter
from prakash.utils import DFUtils, LogUtils
from prakash.plot_utils import CaliUtils

pMesh = Mesh()

input_fibre = 20
output_fibre = 3

min_c = 0
max_c = 65
sweep_v = 20
data_points = 40
read_nums = 10
sleep_t = 0.01
to_plot = True

external_dir = rf'..\Results\calibrate_intf_mzis\{config.time_stamp}'

for i in range(10):
    if i == 0 :
        js = [8]
    elif i == 9:
        js = [0]
    else:
        js = [9-i, 8-i]
    for j in js:
#
# for i in [11]:
#     for j in [10, 11]:

        phaseshifter = (i,j)

        plot_dir = config.home_dir + rf'\data\plots\phaseshifter_{phaseshifter}\{config.time_stamp}'
        LogUtils.log_config(config.time_stamp, log_dir=plot_dir, filename='log')
        logging.info(f'Calibrate phaseshifter {phaseshifter}. Input fibre {input_fibre}, output fibre {output_fibre}')
        logging.info(f'Sweep from {min_c}mA to {max_c}mA, {read_nums} measurements made at each current with {sleep_t}s intervals')

        c_results, pp_popt, pi_popt = pMesh.calibrate_from_sweep(phaseshifter, min_c=min_c, max_c=max_c, sweep_v=sweep_v,
                                                     data_points=data_points, read_nums=read_nums, sleep_t=sleep_t)

        CaliUtils.plot_ps_sweeps(phaseshifter, c_results=c_results, plot_dir=plot_dir, pp_popt=pp_popt,
                                 pi_popt=pi_popt, normalized=True)


        np.save(DFUtils.create_filename(external_dir + rf'\phaseshifter_{phaseshifter}.npy'), c_results)

        mean_results = np.mean(c_results, axis=1)
        std_results = np.std(c_results, axis=1)
        (sweep_cs, mean_cs, mean_vs, electric_power, resistance, optical_power) = tuple(mean_results.T)

        plt.figure('All phaseshifters')
        plt.errorbar(electric_power, optical_power, xerr=std_results[:, 3], yerr=std_results[:, 5], label=f'{phaseshifter}')


pMesh.save_calibration()

powerSupply.zero_all()
powerSupply.close()

plt.figure('All phaseshifters')
plt.xlabel('Measured Electric power (mW)')
plt.ylabel('Optical power (mW)')
plt.legend()
plt.savefig(external_dir + r'\all_phaseshifters.png')

