import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.signal import find_peaks
import logging

from prakash.driver.power_supply import XPOWu
from prakash.driver.power_meter import PM100D
from prakash.mesh import Mesh
from prakash.components import Phaseshifter
from prakash.utils import LogUtils
import prakash.config as config

mesh_name = 'prakash_one'
ps_label = (2,6)


'''Connect devices '''
power_supply = XPOWu(['COM15', 'COM16', 'COM17'])
# Find resource by pyvisa.ResourceManager().list_resources()
opm = PM100D('USB0::0x1313::0x8078::P0022518::INSTR')  # optical power meter
pMesh = Mesh(name=mesh_name, power_supply=power_supply)

targetPS = Phaseshifter(mesh_name, label=ps_label, power_supply=power_supply)
read_num = 10
channel = targetPS.channel

plot_dir = config.home_dir + rf'\..\Results\test_optimisation\ps_{ps_label}_{config.time_stamp}'

LogUtils.log_config(config.time_stamp, log_dir=plot_dir, filename='log_', level=logging.INFO)
logging.info( rf'mesh_name={mesh_name}, ps_label={ps_label}. Test scipy optimisation methods on finding maximum and minimum optical power.')

'''Previous calibration data'''
cal_file = rf'C:\Users\PhotonsLocalAdmin\Documents\prakash_one_control\Results\test_crosstalk\PS_(2, 6)_sweep_crosstalker_2024-03-22(13-10-47.734041)\Sweep_ps(2, 6)_without_crosstalk.npy'
prev_results = np.load(cal_file)
prev_means = np.mean(prev_results, axis=1)
(set_cprev, _, _, epprev, _, opprev) = tuple(prev_means.T)

fig, ax = plt.subplots(layout='constrained')
ax.plot(set_cprev, opprev, '.', markersize=4)
ax.set_xlabel('Set current (mA)')
ax.set_ylabel('Optical power (mW)')

i_min = np.argmin(opprev)
i_peaks, peak_props = find_peaks(opprev, distance=i_min, width=10)  # there may be two

ax.plot(set_cprev[i_min], opprev[i_min], 'r.', markersize=8)
ax.plot(set_cprev[i_peaks], opprev[i_peaks], 'r.', markersize=8)

fig.savefig(plot_dir + rf'\previous_calibration.pdf')

logging.info(f'Use calibration data from {cal_file}. \n'
             f'The minimum from sweep is {np.min(opprev)}mW at set current={set_cprev[i_min]}mA. \n'
             f'The peaks from sweep are at set currents={set_cprev[i_peaks]}mA with optical power= {opprev[i_peaks]}mW.')

'''Find min and max power'''
def probe(current, channel, read_num=10, invert=False):
    current = np.atleast_1d(current)
    current = current[0]

    power_supply.set_current(channel, current)
    sleep(1)

    ops = np.zeros(read_num)
    for i in range(read_num):
        ops[i] = opm.read() * 1000  # mW
        sleep(0.1)

    op = np.mean(ops)
    print(rf'Set {current}mA, read {op}mW')

    if invert:
        return - op
    else:
        return op


def find_minmax(channel, guess_min, guess_max, read_num, method_args=None):
    if method_args is None:
        method_args = {'method': 'Nelder-Mead', 'options': {'xatol': 0.005}}

    power_supply.set_voltage(channel, 15)  #

    min_res = minimize(probe, np.atleast_1d(guess_min), args=(channel, read_num, False), **method_args)
                        # method='Nelder-Mead', options={'xatol': 0.005, 'fatol': 1e-6, 'maxfev':200})  # fatol is 1nW

    power_supply.set_current(channel, 0)

    max_res = minimize(probe, np.atleast_1d(guess_max), args=(channel, read_num, True), **method_args)
                        # method='Nelder-Mead', options={'xatol': 0.005, 'fatol': 0.001, 'maxfev':200})  # need to set a fatol here.

    power_supply.set_channel(channel, 0, 0)

    return min_res, max_res

try:
    guess_min = set_cprev[i_min]
    guess_max = set_cprev[i_peaks[0]]

    bnds = Bounds(0, 65)
    method_args = {'method': 'L-BFGS-B', 'bounds': bnds, 'options': {'ftol': 1e-4, 'maxiter':200}}

    logging.info(f'The method arguments are {method_args}')

    min_res, max_res = find_minmax(channel, guess_min, guess_max, read_num, method_args=method_args)

    logging.info(f'The minimisation result is \n {min_res}')
    logging.info(f'The maximisaiton result is \n {max_res}')

    power_supply.set_channel(channel, 0, 0)
    power_supply.zero_all()

except KeyboardInterrupt:
    print("Shutdown requested... zero all power supply channels")
    power_supply.set_channel(channel, 0, 0)
    power_supply.zero_all()
    raise

except Exception as e:
    print('Ran into exception, zero all power supply channels...')
    power_supply.zero_all()
    raise
