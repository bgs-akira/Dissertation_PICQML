import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import time

from prakash.driver.power_supply import XPOWu
from prakash.driver.power_meter import PM100D
from prakash.mesh import Mesh
from prakash.utils import progressbar, DFUtils
import prakash.config as config

'''Test the response of chip to a sudden or gradual current.'''
mesh_name = 'prakash_one'
ps_label = (2,6)

'''Connect devices '''
power_supply = XPOWu(['COM15', 'COM16', 'COM17'])
# Find resource by pyvisa.ResourceManager().list_resources()
opm = PM100D('USB0::0x1313::0x8078::P0022518::INSTR')  # optical power meter
pMesh = Mesh(name=mesh_name, power_supply=power_supply)
channel = pMesh.get_channel(ps_label)

plot_dir = config.home_dir + rf'\..\Results\test_response\ps_{ps_label}_{config.time_stamp}'

'''Test params'''
test_current = 35
voltage_lim = 15
gradual = np.linspace(5, test_current, 10)

data_separation = 0.1 # seconds
datapoints = 3000

'''Take data'''
sudden_results = np.zeros((datapoints, 4), dtype=float)  # columns=[time, v, c, op]
gradual_results = np.zeros((datapoints, 4), dtype=float)
try:
    '''Sudden'''
    t0 = time.time()
    power_supply.set_channel(channel, voltage_lim, test_current)
    for i in progressbar(range(datapoints), prefix='Read response to sudden current'):
        v,c = power_supply.read_channel(channel)
        op = opm.read()*1000 # unit mW
        sudden_results[i, :] = [time.time() - t0, v, c, op]

        sleep(data_separation)

    power_supply.set_channel(channel, 0, 0)
    power_supply.zero_all()

    for i in progressbar(range(20), prefix=r'Sleeping...'):
        sleep(1)

    '''Gradual'''
    power_supply.set_voltage(channel, voltage_lim)

    t0 = time.time()
    for i_c in progressbar(range(len(gradual)), prefix='Sweep up current gradually'):
        power_supply.set_current(channel, gradual[i_c])
        for i in range(10):
            v, c = power_supply.read_channel(channel)
            op = opm.read()*1000  # unit mW
            gradual_results[i_c * 10 + i, :] = [time.time() - t0, v, c, op]
            sleep(data_separation)
            # should be 1 second in total after each set current

    for i_r in progressbar(range(datapoints- len(gradual) * 10), prefix='Read response to gradual current'):
        v,c = power_supply.read_channel(channel)
        op = opm.read()*1000
        gradual_results[len(gradual)*10 + i_r, :] = [time.time() - t0, v, c, op]

        sleep(data_separation)

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


np.save(DFUtils.create_filename(plot_dir + rf'\gradual_change_of_currents.npy'), gradual)
np.save(plot_dir + rf'\response_to_sudden_{test_current}mA.npy', sudden_results)
np.save(plot_dir + rf'\response_to_gradual_{test_current}mA.npy', gradual_results)

'''Plotting'''
starts = [0,200]
ends = [200, datapoints]

fig, axs = plt.subplots(3,2, sharex='col', layout='constrained', figsize=(12, 8))

for i_col in range(axs.shape[1]):
    start = starts[i_col]
    end = ends[i_col]
    ax1, ax2, ax3 = axs[:, i_col]

    ax1.plot(sudden_results[start:end, 0], sudden_results[start:end, 3], label='Sudden', alpha=0.6)
    ax1.plot(gradual_results[start:end, 0], gradual_results[start:end, 3], label='Gradual', alpha=0.6)
    ax1.set_ylabel('Optical power (mW)')
    ax1.legend()

    ax2.plot(sudden_results[start:end, 0], sudden_results[start:end, 1], label='Sudden', alpha=0.6)
    ax2.plot(gradual_results[start:end, 0], gradual_results[start:end, 1], label='Gradual', alpha=0.6)
    ax2.set_ylabel('Measured voltage (V)')

    ax3.plot(sudden_results[start:end, 0], sudden_results[start:end, 2], label='Sudden', alpha=0.6)
    ax3.plot(gradual_results[start:end, 0], gradual_results[start:end, 2], label='Gradual', alpha=0.6)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Measured current (mA)')

fig.suptitle(f'Response of PS{ps_label} to current')
fig.savefig(plot_dir + rf'\Response_of_ps{ps_label}_to_{test_current}mA.pdf')