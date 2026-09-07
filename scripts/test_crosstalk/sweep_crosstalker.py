import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import logging

from prakash.driver.power_supply import XPOWu
from prakash.driver.power_meter import PM100D
from prakash.components import Phaseshifter
from prakash.calibrator import Calibrator
from prakash.plot_utils import plot_ps_sweeps, plot_mzi_sweeps
from prakash.utils import LogUtils, DFUtils
import prakash.config as config

'''Connect devices '''
mesh_name = 'prakash_one'
power_supply = XPOWu(['COM15', 'COM16', 'COM17'])
# Find resource by pyvisa.ResourceManager().list_resources()
opm = PM100D('USB0::0x1313::0x8078::P0022518::INSTR')  # optical power meter

'''Find target'''
ps_label = (i,j) = (2,6)
targetPS = Phaseshifter(mesh_name, ps_label, power_supply=power_supply)
neighbourPS = Phaseshifter(mesh_name, (i - 1, j), power_supply=power_supply)

plot_dir =config.home_dir + rf'\..\Results\test_crosstalk\PS_{ps_label}_sweep_crosstalker_{config.time_stamp}'

'''Logging'''
LogUtils.log_config(config.time_stamp, log_dir=plot_dir, filename='log_', level=logging.INFO)
logging.info(f'Meshname={mesh_name}. Sweep PS {(i-1,j)} to check the optical power response of light going through the '
             f'target PS {ps_label}.')

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
    'save_data': False
}

logging.info(f'Read number = {read_nums}, sleep_t={sleep_t}, sweep_v={sweep_v}V.')

'''Sweep target phaseshifter first'''
try:
    logging.info(f'Sweep target phaseshifter {targetPS.label} first to calibrate the optical power response. ')
    t_results = targetPS.sweep_current(**sweep_params)

    sleep(1)
    targetPS.zero()
    power_supply.zero_all()
except KeyboardInterrupt:
    print("Shutdown requested... zero all power supply channels")
    power_supply.set_channel(targetPS.channel, 0, 0)
    power_supply.set_channel(neighbourPS.channel, 0, 0)
    power_supply.zero_all()
    raise
except Exception as e:
    print('Ran into exception, zero all power supply channels...')
    power_supply.zero_all()
    raise

np.save(DFUtils.create_filename(plot_dir+rf'\Sweep_ps{targetPS.label}_without_crosstalk.npy'), t_results)
mean_t = np.mean(t_results, axis=1)
(set_ct, _, _ , ept, _, opt) = tuple(mean_t.T)

'''Set up figure'''
fig, axs = plt.subplots(2,2, sharey='all', sharex='col', layout='constrained', figsize=(12, 8))

ax = axs[0,0]
ax.set_title(f'Sweep PS{targetPS.label} without crosstalk')
ax.set_xlabel('Electric power (mW)')
ax.set_ylabel('Optical power (mW)')
ax.plot(ept, opt, '.', markersize=4)

ax = axs[0, 1]
ax.set_xlabel(f'Set current (mA)')
ax.plot(set_ct, opt, '.', markersize=4)

ax1 = axs[1,0]
ax2 = axs[1,1]

ax1.set_title(f'Sweep PS{neighbourPS.label} the crosstalker')
ax1.set_ylabel('Optical power (mW)')
ax1.set_xlabel(f'Electric power (mW)')
ax2.set_xlabel(f'Set current (mA)')

fig.suptitle(f'Sweeping crosstalker')

target_cs = [0, 30]
for target_c in target_cs:

    try:
        if target_c == 0:
            targetPS.set_current(c=0, voltage_lim=0)
        else:
            targetPS.set_current(c=target_c, voltage_lim=sweep_v)

        sleep(2)
        logging.info(f'Set target PS to be c={target_c}. ')

        c_results = neighbourPS.sweep_current(**sweep_params)
        sleep(2)

        targetPS.zero()
        neighbourPS.zero()
    except KeyboardInterrupt:
        print("Shutdown requested... zero all power supply channels")
        power_supply.set_channel(targetPS.channel, 0, 0)
        power_supply.set_channel(neighbourPS.channel, 0, 0)
        power_supply.zero_all()
        raise

    except Exception as e:
        print('Ran into exception, zero all power supply channels...')
        power_supply.zero_all()
        raise

    np.save(DFUtils.create_filename(plot_dir + rf'\{target_c}mA_on_ps{targetPS.label}_while_sweep_ps{neighbourPS.label}.npy'), c_results)

    mean_results = np.mean(c_results, axis=1)
    (set_cs, _, _, eps, _, ops) = tuple(mean_results.T)

    ax1.plot(eps, ops, '.', markersize=4, label=f'{target_c}mA on PS{targetPS.label}')
    ax1.legend()

    ax2.plot(set_cs, ops, '.', markersize=4, label=f'{target_c}mA on PS{targetPS.label}')
    ax2.legend()


fig.savefig(plot_dir + rf'\optical_power_changes_marked.pdf')


