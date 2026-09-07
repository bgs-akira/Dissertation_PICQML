import numpy as np
import pickle
import matplotlib.pyplot as plt
from time import sleep

from prakash.driver.power_supply import XPOWu
from prakash.driver.power_meter import PM100D
from prakash.mesh import Mesh
from prakash.calibrator import Calibrator
from prakash.plot_utils import plot_ps_sweeps, plot_mzi_sweeps
from prakash.utils import progressbar, DFUtils
import prakash.config as config

'''The power meter readings seem to plateau at half-pi peak. '''

mesh_name = 'prakash_one'
mzi_label = (1,7)
i,j = mzi_label

# plot_dir = r'C:\Users\PhotonsLocalAdmin\Documents\prakash_one_control\Results\test_delta_sweep\mzi_(0, 8)_2024-03-06(19-23-17.512744)'
plot_dir =config.home_dir + rf'\..\Results\test_delta_sweep\mzi_{mzi_label}_{config.time_stamp}'
compare_dir = r'C:\Users\PhotonsLocalAdmin\Documents\prakash_one_control\Results\test_delta_sweep\mzi_(1, 7)_2024-03-11(15-11-45.490395)'

'''Connect devices '''
power_supply = XPOWu(['COM15', 'COM16', 'COM17'])
# Find resource by pyvisa.ResourceManager().list_resources()
opm = PM100D('USB0::0x1313::0x8078::P0022518::INSTR')  # optical power meter

pMesh = Mesh(name=mesh_name, power_supply=power_supply)

targetMZI = pMesh.mzi(mzi_label)
targetPS = targetMZI.up_ps
channel = targetPS.channel
'''Sweep parameters'''
sweep_cs = Calibrator.construct_sweep_cs(min_c=20, max_c=25, data_points=100, safe_c_step=2)
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


'''Sweep'''
for c_i in np.linspace(0, 20, 5):
    power_supply.set_channel(channel, v=sweep_v, c=c_i)
    sleep(2)

    v, c = power_supply.read_channel(channel)
    print(f'Set channel={channel} to v={sweep_v}, c={c_i}. Read v={v}, c={c}. ')

c_results = targetPS.sweep_current(**sweep_params)
sleep(5)
np.save(DFUtils.create_filename(plot_dir + rf'\{i,j}_c_results.npy'), c_results)

mean_1 = np.mean(c_results, axis=1)
(set_c1, c1, v1, ep1, _, op1) = tuple(mean_1.T)

c_results_c = np.load(compare_dir + rf'\{i,j}_c_results.npy')
mean_c = np.mean(c_results_c, axis=1)

(set_c2, c2, v2, ep2, _, op2) = tuple(mean_c.T)

'''Plot'''
fig, axs = plt.subplots(2,1, layout='constrained')

ax = axs[0]
ax.plot(ep2, op2, '.', markersize=4, label='previous data')
ax.plot(ep1, op1, '.', markersize=4, label='zoomed in')
ax.set_title(f'Phaseshifter {(i,j)}')
ax.set_xlabel('Electric power (mW)')
ax.set_ylabel('Optical power (mW)')
ax.legend()
ax.set_xlim(ep2[25], ep2[50])
ax.set_ylim(4.1, 4.3)

ax = axs[1]
ax.plot(set_c2, c2, '.', markersize=4, label='previous data')
ax.plot(set_c1, c1, '.', markersize=4, label='zoomed in')
ax.set_xlabel('Set current (mA)')
ax.set_ylabel('Measured current (mA)')
ax.legend()
ax.set_xlim(set_c2[25], set_c2[50])
ax.set_ylim(19, 26)

fig.savefig(plot_dir + rf'\zoomed_in.pdf')