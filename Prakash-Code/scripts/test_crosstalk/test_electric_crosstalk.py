import numpy as np
from time import sleep
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import prakash.config as config
from prakash.driver.optical_switch import LF30CHSM, Hand
from prakash.driver.power_supply import XPOWu
from prakash.driver.power_meter import PM100D
from prakash.mesh import Mesh
from prakash.utils import DFUtils, LogUtils, progressbar

'''Produce a colour map of electrical crosstalk here'''
mesh_name = 'prakash_one'


plot_dir = config.home_dir + rf'\..\Results\test_crosstalk\electric_crosstalk_{config.time_stamp}'

'''Connect to hardware '''
power_supply = XPOWu(['COM15', 'COM16', 'COM17'])
sleep(1)

pMesh = Mesh(name=mesh_name, power_supply=power_supply)

'''Logging'''
LogUtils.log_config(config.time_stamp, log_dir=plot_dir, filename='log_', level=logging.INFO)
logging.info(f'Meshname={mesh_name}. ')

'''Choosing the target'''
ps_label = (4,3)
ch_target = pMesh.get_channel(ps_label)
logging.info(f'Test electric crosstalk by applying current to {ps_label} and measuring all other channels. ')

'''Collect data'''
read_num = 1
sweep_v = 15
sweep_cs = [30, 60]
logging.info(f'Read_num={read_num}, sweep_v={sweep_v}V (this is the set voltage for the target phaseshifter, '
             f'while only the current is varied). sweep_currents = {sweep_cs}')

for sweep_c in sweep_cs:
    voltages = np.zeros((10,10, read_num), dtype=float)
    currents = np.zeros((10,10, read_num), dtype=float)

    try:
        power_supply.set_channel(ch_target, v=sweep_v, c=sweep_c)
        sleep(2)

        v_target, c_target = power_supply.read_channel(ch_target)
        logging.info(f'Set channel {ch_target} v={sweep_v}, c={sweep_c}, measure after 2s v={v_target}, c={c_target}.')

        for i_ps in progressbar(range(100), prefix='Electric readings on phaseshifters'):
            i = i_ps % 10
            j = i_ps // 10
            channel = pMesh.get_channel((i,j))
            for k in range(read_num):
                v, c = power_supply.read_channel(channel)
                voltages[i,j,k] = v
                currents[i,j,k] = c
                sleep(0.1)

        power_supply.set_channel(ch_target, 0, 0)
    except KeyboardInterrupt:
        print("Shutdown requested... zero all power supply channels")
        power_supply.set_voltage(ch_target, 0)
        power_supply.set_current(ch_target, 0)
        power_supply.zero_all()
        raise

    except Exception as e:
        print('Ran into exception, zero all power supply channels...')
        power_supply.zero_all()
        raise

    np.save(DFUtils.create_filename(plot_dir + rf'\heat_on_ps{ps_label}\voltage_when_c={sweep_c:.2f}.npy'), voltages)
    np.save(plot_dir + rf'\heat_on_ps{ps_label}\current_when_c={sweep_c:.2f}.npy', currents)

    '''Plotting'''
    xs = np.zeros(100, dtype=int)  # coordinates in 3d bar plot
    ys = np.zeros(100, dtype=int)
    vs_flat = np.zeros(100, dtype=float)
    e_vs_flat = np.zeros(100, dtype=float)

    for i_ps in range(100):
        i = i_ps % 10
        j = i_ps // 10
        if j <= 4:
            xs[i_ps] = j
            ys[i_ps] = 10 - i
        else:
            xs[i_ps] = 9 - j
            ys[i_ps] = i - 10

        vs_flat[i_ps] = np.mean(voltages[i,j,:])  * 1000  # unit-mV

    bottom = np.zeros(100)
    width= 0.2
    depth=0.9

    cmap = cm.get_cmap('jet')
    color_v = [cmap((np.log10(v_i) - np.log10(10)) / np.log10(10000)) for v_i in vs_flat]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xs, ys, bottom, width, depth, np.log10(vs_flat), color=color_v)

    zticks = np.array([1,2,3,4])
    ax.set_zticks(zticks)
    ax.set_zticklabels([rf'$10^{{{exponent}}}$' for exponent in zticks])
    ax.set_zlim(0, 4)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f'Voltage response to {sweep_c}mA on {ps_label}')
    ax.set_zlabel('Voltage (mV)')

    fig.savefig(plot_dir + rf'\heat_on_ps{ps_label}\voltage_when_c={sweep_c:.2f}.pdf')
