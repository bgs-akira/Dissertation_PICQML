import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import logging
from scipy.optimize import minimize
from scipy.optimize import Bounds

from prakash.driver.power_supply import XPOWu
from prakash.driver.power_meter import PM100D
from prakash.mesh import Mesh
from prakash.calibrator import Calibrator
from prakash.utils import LogUtils
import prakash.config as config

mesh_name = 'prakash_one'
mzi_label = (1, 7)
i, j = mzi_label

plot_dir = config.home_dir + rf'\..\Results\test_delta_sweep\mzi_{mzi_label}_{config.time_stamp}'

'''Connect devices '''
power_supply = XPOWu(['COM15', 'COM16', 'COM17'])
# Find resource by pyvisa.ResourceManager().list_resources()
opm = PM100D('USB0::0x1313::0x8078::P0022518::INSTR')  # optical power meter
pMesh = Mesh(name=mesh_name, power_supply=power_supply)
targetMZI = pMesh.mzi(mzi_label)

if power_supply is not None:
    LogUtils.log_config(config.time_stamp, log_dir=plot_dir, filename='log_', level=logging.INFO)
    logging.info(
        rf'mesh_name={mesh_name}, mzi_label={mzi_label}, use result from previous calibration to initialise b1-b2 and k1, k2.')

'''Read initial data'''
c_results1 = targetMZI.up_ps.read_sweep()
mean_results1 = np.mean(c_results1, axis=1)
(set_c1, c1, v1, ep1, _, op1) = tuple(mean_results1.T)

popt1 = Calibrator.fit_ps_sweeps(c_results1)

c_results2 = targetMZI.low_ps.read_sweep()
mean_results2 = np.mean(c_results2, axis=1)
(set_c2, c2, v2, ep2, _, op2) = tuple(mean_results2.T)

popt2 = Calibrator.fit_ps_sweeps(c_results2)

'''Plot op'''
fig1, axs1 = plt.subplots(2, 1, sharex='col', figsize=(8, 5), layout='constrained')
ax = axs1.flatten()[0]
ax.plot(ep1, op1 / np.max(op1), '.', alpha=0.5, markersize=4, label='data')
ax.set_xlabel('Electric Power (mW)')
ax.set_ylabel('Optical Power (mW)')

ax = axs1.flatten()[1]
ax.plot(ep2, op2 / np.max(op2), '.', alpha=0.5, markersize=4, label='data')
ax.set_xlabel('Electric Power (mW)')
ax.set_ylabel('Optical Power (mW)')

'''Estimate b1-b2 from the initial data'''
p00 = np.mean([op1[0], op2[0]])  # power when no power is applied
pmax = np.mean([np.max(op1), np.max(op2)])  # max power
pmin = np.mean([np.min(op1), np.min(op2)])  # min power

b = np.arcsin((2 * p00 - pmin - pmax) / (pmax - pmin)) - np.pi / 2  # this is always between -pi to 0.
if popt1[-1] - np.pi / 2 > 0:
    b = - b

logging.info(f'Estimating from the pmax and pmin of the previous calibration data, b1-b2={b}')

'''Find min and max power'''
def probe(current, channel, read_num=10, invert=False):
    current = np.atleast_1d(current)
    current = current[0]

    power_supply.set_current(channel, current)
    print(rf'Set current {current}mA')
    sleep(1)

    ops = np.zeros(read_num)
    for i in range(read_num):
        ops[i] = opm.read() * 1000  # mW
        sleep(0.1)

    if invert:
        return - np.mean(ops)
    else:
        return np.mean(ops)

def find_minmax(channel, guess_min, guess_max, read_num):
    bnds = Bounds(0, 65)
    power_supply.set_voltage(channel, 10)#

    min_res = minimize(probe, guess_min, args=(channel, read_num, False),
                        method='Nelder-Mead', bounds=bnds, options={'xatol': 0.005, 'fatol': 1e-6, 'maxfev':200})  # fatol is 1nW
    power_supply.set_current(channel, 0)

    max_res = minimize(probe, guess_max, args=(channel, read_num, True),
                        method='Nelder-Mead', bounds=bnds,
                        options={'xatol': 0.005, 'fatol': 0.001, 'maxfev':200})  # need to set a fatol here.

    power_supply.set_current(channel, 0)
    power_supply.set_voltage(channel, 0)

    return min_res, max_res

try:
    read_num = 10

    # find min/max for up_ps
    channel1 = targetMZI.up_ps.channel
    guess_min1 = set_c1[np.argmin(op1)]
    guess_max1 = set_c1[np.argmax(op1)]

    min_res1, max_res1 = find_minmax(channel1, guess_min1, guess_max1, read_num)

    logging.info(f'Run scipy optimise to find min and max op for phaseshifter {targetMZI.up_ps.label}. \n'
                 f'Minimise result is {min_res1}. \n'
                 f'Maximise result is {max_res1}')

    pmin1 = min_res1.fun
    pmax1 = - max_res1.fun

    # find min/max for low_ps
    channel2 = targetMZI.low_ps.channel
    guess_min2 = set_c2[np.argmin(op2)]
    guess_max2 = set_c2[np.argmax(op2)]

    min_res2, max_res2 = find_minmax(channel2, guess_min2, guess_max2, read_num)

    logging.info(f'Run scipy optimise to find min and max op for phaseshifter {targetMZI.low_ps.label}. \n'
                 f'Minimise result is {min_res2}. \n'
                 f'Maximise result is {max_res2}')

    pmin2 = min_res2.fun
    pmax2 = - max_res2.fun

    power_supply.zero_all()

except KeyboardInterrupt:
    print("Shutdown requested... zero all power supply channels")
    power_supply.zero_all()
    raise

except Exception as e:
    print('Ran into exception, zero all power supply channels...')
    power_supply.zero_all()
    raise


'''Estimate b from the new min max estimates'''
p00 = opm.read() * 1000  # everything is zeroed now.
pmax = np.mean([pmax1, pmax2])  # max power
pmin = np.mean([pmin1, pmin2])  # min power

b = np.arcsin((2 * p00 - pmin - pmax) / (pmax - pmin)) - np.pi / 2  # this is always between -pi to 0.
if popt1[-1] - np.pi / 2 > 0:
    b = - b

logging.info(f'From the new pmin/pmax readings, new estimate of b1-b2 = {b}')



# '''Construct new sweep_cs  - doesn't seem to be useful. '''
# def new_sweep_cs(phaseshifter):
#     sweep_cs = Calibrator.construct_sweep_cs(0, 65, 100)
#
#     c_results = phaseshifter.read_sweep()
#     mean_results = np.mean(c_results, axis=1)
#     (set_c, _, _, ep, _, _) = tuple(mean_results.T)
#
#     popt = Calibrator.fit_ps_sweeps(c_results)
#     k = popt[-2]
#
#     start_c1 = set_c[np.argmax(ep > (np.pi/4 - b) / k)]
#     stop_c1 = set_c[np.argmax(ep > (3*np.pi/4 - b) / k)]
#     start_1 = np.argmax(sweep_cs > start_c1)
#     stop_1 = np.argmax(sweep_cs > stop_c1)
#
#     fill1 = np.arange(sweep_cs[start_1], sweep_cs[stop_1], 0.05)
#
#     start_c2 = set_c1[np.argmax(ep > (5*np.pi/4 - b) / k)]
#     stop_c2 = set_c1[np.argmax(ep > (7*np.pi/4 - b) / k)]
#     start_2 = np.argmax(sweep_cs > start_c2)
#     stop_2 = np.argmax(sweep_cs > stop_c2)
#
#     fill2 = np.arange(sweep_cs[start_2], sweep_cs[stop_2], 0.05)
#
#     return np.concatenate([sweep_cs[:start_1], fill1, sweep_cs[stop_1: start_2], fill2, sweep_cs[stop_2:]])
#
# sweep_cs1 = new_sweep_cs(targetMZI.up_ps)
# sweep_cs2 = new_sweep_cs(targetMZI.low_ps)
#
# fig2, axs2 = plt.subplots(2, 1, sharex='all', layout='constrained')
# ax = axs2[0]
# ax.plot(np.arange(len(sweep_cs1)), sweep_cs1, '.', markersize=4)
# ax.set_ylabel('Set current (mA)')
#
# ax = axs2[1]
# ax.plot(np.arange(len(sweep_cs2)), sweep_cs2, '.', markersize=4)
# ax.set_ylabel('Set current (mA)')
#
#
# '''Sweep upper PS on the new currents'''
# new_results1 = targetMZI.sweep_up(sweep_cs1, sweep_v=10, read_nums=1, opm=opm)
# np.save(plot_dir + rf'\{targetMZI.up_ps.label}_c_results.npy', new_results1)
#
# new_mean1 = np.mean(new_results1, axis=1)
# p0 = popt1
# p0[-1] = np.pi/2 + b
#
# new_popt1 = Calibrator.fit_ps_sweeps(new_results1, p0=p0)
#
# fig1, axs1 = plot_ps_sweeps((i,j), mesh_name, c_results=new_results1,
#                             pp_popt=new_popt1, plot_dir=plot_dir)
#
#
# '''Sleep'''
# for i_s in progressbar(range(5), prefix=f'Sleep 5s'):
#     sleep(1)
#
# print(rf'Read up_ps channel: {power_supply.read_channel(targetMZI.up_ps.channel)}')
# print(rf'Read low_ps channel: {power_supply.read_channel(targetMZI.low_ps.channel)}')
#
#
# '''Sweep lower PS'''
# new_results2 = targetMZI.sweep_low(sweep_cs2, sweep_v=10, read_nums=1, opm=opm)
# np.save(plot_dir + rf'\{targetMZI.low_ps.label}_c_results.npy', new_results2)
#
# new_mean2 = np.mean(new_results2, axis=1)
#
# p0 = popt2
# p0[-1] = np.pi/2 - b
# new_popt2 = Calibrator.fit_ps_sweeps(new_results2, p0=p0)
#
# fig2, axs2 = plot_ps_sweeps((i+1,j), mesh_name, c_results=new_results2,
#                             pp_popt=new_popt2, plot_dir=plot_dir)

