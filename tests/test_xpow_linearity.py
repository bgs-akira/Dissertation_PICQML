import numpy as np
import json
from scipy.stats import linregress
import matplotlib.pyplot as plt
from time import sleep
import pandas as pd

from prakash.driver.power_supply import XPOWu
from prakash.driver.power_meter import PM100D
from prakash.mesh import Mesh
from prakash.calibrator import Calibrator
from prakash.plot_utils import plot_ps_sweeps, plot_mzi_sweeps
from prakash.utils import progressbar, DFUtils
import prakash.config as config

channel = 87
time_stamp = config.time_stamp
power_supply = XPOWu(['COM15', 'COM16', 'COM17'])

#TODO: write a test script that sweeps current on external resistor, and log down XPOW measurement and indpendent multimeter measurements. Plot and check linearity.

sweep_cs = Calibrator.construct_sweep_cs(min_c=0, max_c=65, data_points=300, safe_c_step=2)
sweep_v = 15

results_df = pd.DataFrame(columns=['sweep_v', 'sweep_c', 'xpow_v', 'xpow_c', 'meter_v', 'meter_c'])
results_dir = rf'..\Results\test_xpow_linearity\channel_{channel}_{time_stamp}'

prev_val2 = 0.
prev_val = 0.
try:
    power_supply.set_voltage(channel, sweep_v)
    print(f'Setting channel {channel} voltage={sweep_v}')

    read_type = input('What is your input? [meter_v/meter_c]')
    for i_c, sweep_c in enumerate(sweep_cs):

        power_supply.set_current(channel, sweep_c)
        print(f'Setting channel {channel} current={sweep_c}')
        sleep(1)

        v, c = power_supply.read_channel(channel)

        read_val = float(input('Input value? (Unit: V or mA)'))
        if read_val <= prev_val or read_val - prev_val >= prev_val - prev_val2:
            read_val = float(input('Please confirm the value:'))

        results_df.loc[i_c] = {
            'sweep_v': sweep_v, 'sweep_c': sweep_c, 'xpow_v': v, 'xpow_c': c, read_type: read_val
        }

        prev_val2 = prev_val
        prev_val = read_val

    power_supply.set_channel(channel, 0, 0)
    power_supply.zero_all()

except KeyboardInterrupt:
    print("Shutdown requested... zero all power supply channels")
    power_supply.set_voltage(channel, 0)
    power_supply.set_current(channel, 0)
    power_supply.zero_all()
    raise

except Exception as e:
    print('Ran into exception, zero all power supply channels...')
    power_supply.zero_all()
    raise

results_df.to_csv(DFUtils.create_filename(results_dir + rf'\results_df.csv'))
xpow_cs = np.array(results_df['xpow_c'])
xpow_vs = np.array(results_df['xpow_v'])
meter_vs = np.array(results_df['meter_v'])
meter_cs = np.array(results_df['meter_c'])

if read_type == 'meter_v':
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(10,8), layout='constrained')

    ax = axs[0]
    ax.plot(xpow_cs, xpow_vs, '.', label='By XPOW', alpha=0.5)
    ax.plot(xpow_cs, meter_vs, label='By multimeter')

    ax.legend()
    ax.set_xlabel('XPOW measured current (mA)')
    ax.set_ylabel('Measured voltage (V)')

    ax=axs[1]
    ax.plot(xpow_cs, meter_vs - xpow_vs, label='meter - xpow')
    ax.set_xlabel('XPOW measured current (mA)')
    ax.set_ylabel('Measured voltage (V)')
    ax.legend()

    fig.savefig(results_dir + rf'\voltage_readings_combined.pdf')

    xpow_regress = linregress(xpow_cs, xpow_vs)
    meter_regress = linregress(xpow_cs, meter_vs)

    voltage_regress = linregress(xpow_vs, meter_vs)

    read_r = float(input('Meter resistance reading? (Unit: Ohms)'))

    regress_results = {
        'xpow_v_to_c_slope': xpow_regress.slope,
        'xpow_v_to_c_intercept': xpow_regress.intercept,
        'meter_v_to_xpow_c_slope': meter_regress.slope,
        'meter_v_to_xpow_c_intercept': meter_regress.intercept,
        'meter_v_to_xpow_v_slope': voltage_regress.slope,
        'meter_v_to_xpow_v_intercept': voltage_regress.intercept,
        'meter_read_resistance': read_r
    }

    with open(results_dir + rf'\regress_results.json', 'w') as fp:
        json.dump(regress_results, fp)

elif read_type == 'meter_c':
    fig, axs = plt.subplots(2,1, sharex='all', figsize=(10,8), layout='constrained')

    ax = axs[0]
    ax.plot(sweep_cs, results_df['xpow_c'], label='By XPOW')
    ax.plot(sweep_cs, results_df['meter_c'], '.', label='By multimeter', alpha=0.5)

    ax.legend()
    ax.set_xlabel('Set Current (mA)')
    ax.set_ylabel('Measured Current (mA)')

    ax = axs[1]
    ax.plot(sweep_cs, meter_cs - xpow_cs, label='meter - xpow')

    ax.set_xlabel('Set current (mA)')
    ax.set_ylabel('Measured current (mA)')
    ax.legend()

    fig.savefig(results_dir + rf'\current_readings_combined.pdf')

    meter_regress = linregress(xpow_cs, meter_cs)

    xpow_to_sweep_regress = linregress(sweep_cs, xpow_cs)
    meter_to_sweep_regress = linregress(sweep_cs, meter_cs)

    read_r = float(input('Meter resistance reading? (Unit: Ohms)'))

    regress_results = {
        'meter_c_to_xpow_c_slope': meter_regress.slope,
        'meter_c_to_xpow_c_intercept': meter_regress.intercept,
        'xpow_c_to_sweep_c_slope': xpow_to_sweep_regress.slope,
        'xpow_c_to_sweep_c_intercept': xpow_to_sweep_regress.intercept,
        'meter_c_to_sweep_c_slope': meter_to_sweep_regress.slope,
        'meter_c_to_sweep_c_intercept': meter_to_sweep_regress.intercept,
        'meter_read_resistance': read_r
    }

    with open(results_dir + rf'\regress_results.json', 'w') as fp:
        json.dump(regress_results, fp)