import numpy as np
import matplotlib.pyplot as plt

external_dir = r'../Results/calibrate_intf_mzis/2023-10-05(18-37-14.777541)'

plt.figure('All phaseshifters')
for i in range(10):
    if i == 0 :
        js = [8]
    elif i == 9:
        js = [0]
    else:
        js = [9-i, 8-i]
    for j in js:

        phaseshifter = (i,j)

        c_results = np.load(external_dir + rf'\phaseshifter_{phaseshifter}.npy')

        mean_results = np.mean(c_results, axis=1)
        std_results = np.std(c_results, axis=1)
        (sweep_cs, mean_cs, mean_vs, electric_power, resistance, optical_power) = tuple(mean_results.T)


        plt.errorbar(electric_power, optical_power, xerr=std_results[:, 3], yerr=std_results[:, 5], label=f'{phaseshifter}')


plt.xlabel('Measured Electric power (mW)')
plt.ylabel('Optical power (mW)')

plt.savefig(external_dir + r'\all_phaseshifters_without_legend.png')
