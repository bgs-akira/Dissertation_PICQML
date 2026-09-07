import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import logging

import sys, traceback

from prakash.driver.power_supply import XPOW
from prakash.utils import DFUtils, LogUtils

if __name__ == "__main__":
    # <<<<<<<<<<<<<<<<<<< Setup  >>>>>>>>>>>>>>>>>>
    channel = 60
    resistance = 150
    V_in = 10

    sleep_t_1 = 0.1
    sleep_t_2 = 1.5

    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
    results_dir = rf'..\Results\test_xpow_channels\channel_{channel}_{time_stamp}'

    # <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
    LogUtils.log_config(time_stamp=time_stamp, dir=results_dir, filehead='log', module_name='', level=logging.INFO)
    logging.info(f'Sweep current on resistor {resistance}Ohms. Input Voltage = {V_in}V'
                 f'Change to nicslab voltage and current scaling relationship.')

    # <<<<<<<<<<<<<<<<<<< Connect Device  >>>>>>>>>>>>>>>>>>
    xpow = XPOW(['COM8', 'COM9', 'COM10'])
    msg = 'Device connected'
    logging.info(msg)

    sleep(1)

    try:
        # <<<<<<<<<<<<<<<<<<< Sweep Current  >>>>>>>>>>>>>>>>>>
        sweep_cs = np.arange(start=0, stop=60, step=3)
        c_results = np.zeros((len(sweep_cs), 3), dtype=float)
        c_results[:, 0] = sweep_cs

        voltage_val = 9
        xpow.set_voltage(channel, voltage_val)
        sleep(1)
        initial_v, initial_c = xpow.read_channel(channel)
        logging.info(f'channel={channel}, initial set {voltage_val}V, initial read {initial_v}V and {initial_c}mA')

        for id, sweep_c in enumerate(sweep_cs):
            xpow.set_current(channel, sweep_c)
            sleep(sleep_t_1)
            v_1, c_1 = xpow.read_channel(channel)

            sleep(sleep_t_2)
            v_2, c_2 = xpow.read_channel(channel)

            logging.info(f'channel={channel}, set {sweep_c}mA, after {sleep_t_1}s read {v_1}V and {c_1}mA, '
                         f'after {sleep_t_2}s read {v_2}V and {c_2}mA. Current diff is {c_2 - c_1}mA.')

            c_results[id, 1] = v_1
            c_results[id, 2] = c_1

        xpow.set_voltage(channel, 0)
        xpow.set_current(channel, 0)
        xpow.zero_all()

        c_df = pd.DataFrame(data=c_results, columns=['set_c', 'actual_v', 'actual_c'])
        c_df.to_csv(DFUtils.create_filename(results_dir + r'\sweep_c.csv'))

        plt.figure('actual_c vs set_c')
        plt.plot(sweep_cs, c_results[:, 2], 'x', label='sweep')

        plt.xlabel('Set Current (mA)')
        plt.ylabel('Actual Current (mA)')
        plt.legend()
        plt.title('Sweep current')

        plt.savefig(results_dir + r'\sweep_c.png')

        plt.figure('actual_v vs actual_c')
        plt.plot(c_results[:, 2], c_results[:, 1], 'x', label='sweep')

        # linear regression
        x = c_results[:, 2]
        y = c_results[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        plt.plot(x, m*x+c, 'r', label=f'{m:.3f}*I+({c:.3f})')

        plt.xlabel('Actual Current (mA)')
        plt.ylabel('Actual Voltage (V)')
        plt.legend()
        plt.title(f'Channel{channel} I-V curve')
        plt.savefig(results_dir + r'\IV_curve.png')



    except KeyboardInterrupt:

        print("Shutdown requested... zero xpow channels")
        xpow.zero_all()
        print("Exiting...")

    except Exception:
        xpow.zero_all()

        traceback.print_exc(file=sys.stdout)



