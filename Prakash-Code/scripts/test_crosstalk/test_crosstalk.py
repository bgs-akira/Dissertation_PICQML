import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
from time import sleep
import logging

import prakash.config as config
from prakash.driver.optical_switch import LF30CHSM, Hand
from prakash.driver.power_supply import XPOWu
from prakash.driver.power_meter import PM100D
from prakash.mesh import Mesh
from prakash.calibrator import Calibrator
from prakash.plot_utils import plot_ps_sweeps, plot_mzi_sweeps, plot_sigma_sweeps
from prakash.utils import DFUtils, LogUtils

mesh_name = 'prakash_one'
mzi_label = (2,6)
i, j = mzi_label

neighbour = (0,6)  # in neighbouring mzi or in same mzi

plot_dir = config.home_dir + rf'\..\Results\test_crosstalk\mzi_{mzi_label}_{config.time_stamp}'
data_prev_file = r'/Results/test_crosstalk/mzi_(2, 6)_2024-03-19(17-21-48.459991)/(2, 6)_c_results_when_(1, 6)=0.00mA.npy'
# data_prev_file = None

'''Connect to hardware '''
power_supply = XPOWu(['COM15', 'COM16', 'COM17'])
# Find resource by pyvisa.ResourceManager().list_resources()
opm = PM100D('USB0::0x1313::0x8078::P0022518::INSTR')  # optical power meter
osw = Hand()
sleep(1)

pMesh = Mesh(name=mesh_name, power_supply=power_supply)
targetMZI = pMesh.mzi(mzi_label)
ch1 = targetMZI.up_ps.channel
ch_neighbour = pMesh.get_channel(neighbour) # targetMZI.low_ps.channel


'''Logging'''
LogUtils.log_config(config.time_stamp, log_dir=plot_dir, filename='log_', level=logging.INFO)
logging.info(f'Meshname={mesh_name}. '
             f'Test crosstalk by sweeping phaseshifter {(i,j)} (channel {ch1})'
             f' with heat applied to phaseshifter {neighbour} (channel {ch_neighbour}).')


'''Load in previous calibration data'''
if data_prev_file is not None:
    logging.info(rf'Load zero crosstalk data from {data_prev_file}')
    c_results_prev = np.load(data_prev_file)
    mean_prev = np.mean(c_results_prev, axis=1)
    (set_cprev, cprev, vprev, epprev, _, opprev) = tuple(mean_prev.T)


'''Sweep parameters'''
sweep_cs = Calibrator.construct_sweep_cs(min_c=0, max_c=65, data_points=300, safe_c_step=0.5)
read_nums = 1
sleep_t = 0.1
sweep_v = 15
sweep_params = {
    'sweep_cs': sweep_cs,
    'sweep_v': sweep_v,
    'read_nums': read_nums,
    'sleep_t': sleep_t,
    'opm': opm,
}

'''Set up the plots'''
# sweep results
fig1, ax1 = plt.subplots(figsize=(10, 6), layout='constrained')
# ax1.plot(epprev, opprev, '.', markersize=4, label=r'$c_2=0$mA', alpha=0.5)
ax1.set_xlabel('Electric heat (mW)')
ax1.set_ylabel('Optical power (mW)')
ax1.set_title(f'Phaseshifter {(i,j)} with heat on {neighbour}')

fig2, axs2 = plt.subplots(2, 1, figsize=(10,8), layout='constrained')
ax = axs2[0]
ax.set_ylabel('Optical power (mW)')
ax.set_title(f'Phaseshifter {(i,j)} with heat on {neighbour}')

ax = axs2[1]
ax.set_xlabel('Set current (mA)')
ax.set_ylabel('op - op(c2=0) (mW)')

# electric properties of target
fig3, axs3 = plt.subplots(3,1, figsize=(8, 9), layout='constrained', sharex='all')

ax = axs3[0]
# ax.plot(set_cprev, vprev, alpha=0.8, label=rf'$c_2=0$mA')
ax.set_ylabel('Voltage difference (V)')
ax.set_title(f'Electric properties of PS{(i,j)} under crosstalk')

ax = axs3[1]
# ax.plot(set_cprev, cprev, alpha=0.8, label=rf'$c_2=0$mA')
ax.set_ylabel('Current difference (mA)')

ax = axs3[2]
# ax.plot(set_cprev, epprev, alpha=0.8, label=rf'$c_2=0$mA')
ax.set_ylabel('Electric power difference (mW)')
ax.set_xlabel('Set current (mA)')

'''Set neighbour to some current '''
# c2_halfpi = np.interp(np.pi / 2, low_cal[1] - low_cal[1,0], low_cal[0])
for set_c_neighbour in [30, 45, 60]:

    try:
        power_supply.set_channel(ch_neighbour, v=sweep_v, c=set_c_neighbour)
        sleep(3)
        v_neighbour, c_neighbour = power_supply.read_channel(ch_neighbour)
        v_target, c_target = power_supply.read_channel(ch1)

        logging.info(f'Set phaseshifter {neighbour}, channel={ch_neighbour} to v={sweep_v}V, c={set_c_neighbour}mA. '
                     f'Measure v={v_neighbour}V, c={c_neighbour}mA on channel={ch_neighbour}. '
                     f'Measure v={v_target}V, c={c_target} on channel={ch1}, which is the target ps. ')

        '''Sweep theta1'''
        c_results = targetMZI.sweep_up(**sweep_params)

        power_supply.zero_all()
        sleep(5)

    except KeyboardInterrupt:
        print("Shutdown requested... zero all power supply channels")
        power_supply.zero_all()
        raise

    except Exception as e:
        print('Ran into exception, zero all power supply channels...')
        power_supply.zero_all()
        raise

    np.save(DFUtils.create_filename(plot_dir + rf'\{(i,j)}_c_results_when_{neighbour}={set_c_neighbour:.2f}mA.npy'), c_results)

    # c_results1 = np.load(plot_dir + rf'\{(i,j)}_c_results_when_{neighbour}={c_neighbour:.2f}mA.npy')

    mean_results = np.mean(c_results, axis=1)
    # std_results = np.std(c_results, axis=1)
    (set_cs, mean_cs, mean_vs, eps, _, ops) = tuple(mean_results.T)

    if set_c_neighbour == 0:
        (set_cprev, cprev, vprev, epprev, _, opprev) = tuple(mean_results.T)


    '''Plotting results'''
    plot_params = {'label': rf'$c_2={{{set_c_neighbour}}}$mA', 'alpha':0.5}

    # plot sweep results
    # ax1.errorbar(eps, ops, xerr=std_results[:, 3], yerr=std_results[:, 5], label=rf'$V_2I_2={{{v_neighbour*c_neighbour:.2f}}}$mW', alpha=0.5)
    ax1.plot(eps, ops, **plot_params)
    ax1.legend()

    # optical phase against set current
    ax = axs2[0]
    ax.plot(set_cs, ops, **plot_params)
    ax.legend()

    ax = axs2[1]
    ax.plot(set_cs, ops - opprev, **plot_params)


    # electric measurements of target
    ax = axs3[0]
    ax.plot(set_cs, mean_vs - vprev, **plot_params)
    ax.legend()
    ax = axs3[1]
    ax.plot(set_cs, mean_cs - cprev, **plot_params)
    ax = axs3[2]
    ax.plot(set_cs, eps - epprev, **plot_params)




fig1.savefig(plot_dir + rf'\sweep_{(i,j)}_with_heat_on_{neighbour}.pdf')
fig2.savefig(plot_dir + rf'\sweep_{(i,j)}_with_heat_on_{neighbour}_against_c.pdf')
fig3.savefig(plot_dir + rf'\Electric_diffs_of_ps{(i,j)}_with_heat_on_{neighbour}.pdf')