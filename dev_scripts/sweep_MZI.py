import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import logging

import sys, traceback

import prakash.config as config
from dev_scripts.devices import powerSupply, opticalPowerMeter
from prakash.utils import DFUtils, LogUtils

if __name__ == "__main__":
    # <<<<<<<<<<<<<<<<<<< Setup  >>>>>>>>>>>>>>>>>>
    channel = 59

    laser_power = 12
    input_fibre = 5
    output_fibre = 7
    fan_power = 3.07
    tec_temp = 28
    supply_voltage = 30

    sleep_t_1 = 0.5
    sleep_t_2 = 0.1
    num_measurement = 1

    time_stamp = config.time_stamp
    results_dir = rf'..\Results\sweep_test_MZI\channel_{channel}_in_{input_fibre}_out_{output_fibre}_{time_stamp}'

    # <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
    LogUtils.log_config(time_stamp=time_stamp, log_dir=results_dir, filename='log', level=logging.INFO)
    logging.info(f'Sweep current on test MZI via xpow channel={channel}.\n'
                 f'NKT laser ({laser_power}mW) sent into fibre {input_fibre}, and optical power meter measures fibre {output_fibre}.\n'
                 f'Fan set at {fan_power}V. TEC set temperature {tec_temp} degrees. \n'
                 f'Supply voltage {supply_voltage}V.')


    # <<<<<<<<<<<<<<<<<<< Connect Device  >>>>>>>>>>>>>>>>>>


    wavelength = 1550
    opticalPowerMeter.set_wav(wavelength)

    initial_power = opticalPowerMeter.read()
    logging.info(f'Initial optical power = {initial_power}W')

    try:
        # <<<<<<<<<<<<<<<<<<< Sweep Current  >>>>>>>>>>>>>>>>>>
        sweep_cs1 = np.linspace(0, 20, num=4)
        sweep_cs2 = np.sqrt(np.linspace(20**2, 40**2, num=11))

        sweep_cs = np.concatenate([sweep_cs1, sweep_cs2])
        c_results = np.zeros((len(sweep_cs), num_measurement, 4), dtype=float)

        voltage_val = 20
        powerSupply.set_voltage(channel, voltage_val)
        sleep(sleep_t_1)
        initial_v, initial_c = powerSupply.read_channel(channel)
        logging.info(f'channel={channel}, initial set {voltage_val}V, initial read {initial_v}V and {initial_c}mA')

        for id, sweep_c in enumerate(sweep_cs):
            powerSupply.set_current(channel, sweep_c)
            sleep(sleep_t_1)
            logging.info(f'Set {sweep_c}mA, sleep {sleep_t_1}s')
            for i_meas in range(num_measurement):
                v, c = powerSupply.read_channel(channel)
                optical_power = opticalPowerMeter.read()
                electric_power = v * c
                c_results[id, i_meas, :] = np.array([v, c, optical_power, electric_power])

                sleep(sleep_t_2)

                logging.info(f'Read {v}V, {c}mA, optical {optical_power*1000 :.4f}mW, electric {electric_power}mW, '
                             f'and sleep {sleep_t_2}.')

        powerSupply.set_voltage(channel, 0)
        powerSupply.set_current(channel, 0)
        powerSupply.zero_all()

    except KeyboardInterrupt:

        print("Shutdown requested... zero xpow channels")
        powerSupply.set_voltage(channel, 0)
        powerSupply.set_current(channel, 0)
        powerSupply.zero_all()
        print("Exiting...")

    except Exception:
        powerSupply.zero_all()
        traceback.print_exc(file=sys.stdout)

    np.save(DFUtils.create_filename(results_dir + r'\results.npy'), c_results)
    np.save(results_dir + r'\currents.npy', sweep_cs)

    # <<<<<<<<<<<<<<<<<<< Plot actual current vs set current  >>>>>>>>>>>>>>>>>>
    mean_c = np.mean(c_results[:, :, 1], axis=1)
    std_c = np.std(c_results[:, :, 1], axis=1)

    plt.figure('actual_c vs set_c')
    plt.errorbar(sweep_cs, mean_c, xerr=std_c, fmt='x', linestyle='None')

    plt.xlabel('Set Current (mA)')
    plt.ylabel('Mean Actual Current (mA)')
    plt.title('Sweep current')

    plt.savefig(results_dir + r'\sweep_c.png')

    # <<<<<<<<<<<<<<<<<<< Plot I-V curve  >>>>>>>>>>>>>>>>>>
    mean_v = np.mean(c_results[:, :, 0], axis=1)
    std_v = np.std(c_results[:, :, 0], axis=1)

    plt.figure('actual_v vs actual_c')
    plt.errorbar(mean_c, mean_v, xerr=std_c, yerr=std_v, fmt='x', linestyle='None', label='mean')

    # linear regression
    A = np.vstack([mean_c, np.ones(len(mean_c))]).T
    m, c = np.linalg.lstsq(A, mean_v, rcond=None)[0]
    plt.plot(mean_c, m * mean_c + c, 'r', label=f'{m:.3f}*I+({c:.3f})')

    plt.xlabel('Mean Actual Current (mA)')
    plt.ylabel('Mean Actual Voltage (V)')
    plt.legend()
    plt.title(f'Channel{channel} I-V curve')
    plt.savefig(results_dir + r'\IV_curve.png')

    # <<<<<<<<<<<<<<<<<<< Plot Power vs Current curve   >>>>>>>>>>>>>>>>>>
    mean_optical = np.mean(1000 * c_results[:, :, 2], axis=1)
    std_optical = np.std(1000 * c_results[:, :, 2], axis=1)

    plt.figure('optical power vs actual c')
    plt.errorbar(mean_c, mean_optical, xerr=std_c, yerr=std_optical, fmt='x', linestyle='None')

    plt.xlabel('Mean Actual Current (mA)')
    plt.ylabel('Optical Power (W)')
    plt.title(f'Power on fibre {output_fibre}')

    plt.savefig(results_dir + r'\P_vs_current.png')

    plt.figure('optical power vs set c')
    plt.errorbar(sweep_cs, mean_optical, yerr=std_optical, fmt='x', linestyle='None')

    plt.xlabel('Set Current (mA)')
    plt.ylabel('Optical Power (W)')
    plt.title(f'Power on fibre {output_fibre}')

    plt.savefig(results_dir + r'\P_vs_set_current.png')


    # <<<<<<<<<<<<<<<<<<< Plot Optical Power vs Electric Power curve   >>>>>>>>>>>>>>>>>>
    mean_electric = np.mean(c_results[:, :, 3], axis=1)
    std_electric = np.std(c_results[:, :, 3], axis=1)

    plt.figure('optical power vs electric power')
    plt.errorbar(mean_electric, mean_optical,
                 xerr=std_electric, yerr=std_optical, fmt=',', linestyle='None')

    plt.xlabel('Mean Electric Power (mW)')
    plt.ylabel('Mean Optical Power (mW)')
    plt.title(f'Power on fibre {output_fibre}')

    plt.savefig(results_dir + r'\P_vs_P.png')


