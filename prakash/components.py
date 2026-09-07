import numpy as np
import pandas as pd
import json
import os
from time import sleep
import logging
import warnings

import prakash.config as config
from prakash.utils import DFUtils, progressbar


class Component(object):

    def __init__(self, mesh_name: str, label: tuple, power_supply=None):
        self.mesh = mesh_name
        self.label = label

        self.power_supply = power_supply

    def update_power_supply(self, power_supply):
        self.power_supply = power_supply

class Phaseshifter(Component):

    def __init__(self, mesh_name: str, label: tuple, power_supply=None, channel=None):
        super().__init__(mesh_name, label, power_supply)

        (i, j) = label
        if channel is None:
            df = pd.read_csv(config.home_dir + rf'\{mesh_name}\phaseshifters.csv')
            channels = list(df.loc[(df['i'] == i) & (df['j'] == j)]['Channel'])
            if len(channels) ==0:
                raise ValueError('Channel not found')
            elif len(channels) >1:
                raise ValueError(f'{len(channels)} channels found for said phaseshifter')
            else:
                self._channel = channels[0]
        else:
            self._channel = channel

        # raw file to save sweep results
        self.sweep_file = DFUtils.create_filename(config.home_dir + rf'\{mesh_name}\ps_sweep\{self.label}.npy')

        # Data for electric power interpolation
        self._interp_data = None
        if os.path.isfile(self.sweep_file):
            self.read_interp()

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, new_channel):
        self._channel = new_channel

    @property
    def interp_data(self):
        return (self._interp_data[0].copy(), self._interp_data[1].copy())

    @interp_data.setter
    def interp_data(self, new_interp_data):
        self._interp_data = new_interp_data

    def read_sweep(self, sweep_file=None):
        if sweep_file is None:
            sweep_file = self.sweep_file
        if os.path.isfile(sweep_file):
            c_results = np.load(sweep_file)
        else:
            raise FileNotFoundError(rf'No sweep data found for Phaseshifter {self.label}')

        return c_results

    def read_interp(self, sweep_file=None):
        """Read current and electric power data for interpolation"""
        c_results = self.read_sweep(sweep_file)
        mean_results = np.mean(c_results, axis=1)
        sweep_cs = mean_results[:, 0]
        electric_power = mean_results[:, 3]

        self.interp_data = (sweep_cs, electric_power)

    def set_power(self, power, voltage_lim):
        """
        Set power.
        :param power: Electric power (mW)
        :param voltage_lim: Voltage limit (V)
        """
        try:
            power0 = self.read_electric_power()
            current_val = self.interpolate_current_from_power(power)
            if current_val == 0:
                self.power_supply.set_voltage(self.channel, 0)
            else:
                self.power_supply.set_voltage(self.channel, voltage_lim)
            self.power_supply.set_current(self.channel, current_val)

            # TODO: Sleep for longer if power drop is high. Come up with a better way.
            if power < power0 - 100:
                sleep(2)

        except Exception as e:
            print('Ran into exception, zero all power supply channels...')
            self.power_supply.zero_all()
            raise

    def interpolate_current_from_power(self, power):
        """
        Given a power (mW), interpolate to find what current to apply.  """

        if self._interp_data is None:
            if not os.path.isfile(self.sweep_file):
                raise Exception(
                    rf'No sweep data for Phaseshifter {self.label}. Cannot interpolate to set electric power. ')
            else:
                self.read_interp()

        sweep_cs, electric_power = self._interp_data

        if power >= np.max(electric_power):
            warnings.warn(rf'Interpolation limited at power={np.max(electric_power)}.')
        if power < 0:
            raise ValueError('Power cannot be negative')

        return np.interp(power, electric_power, sweep_cs, left=0.)

    def zero(self):
        """Zero the electric power"""
        self.power_supply.set_voltage(self.channel, 0)
        self.power_supply.set_current(self.channel, 0)

    def sweep_current(self, sweep_cs, sweep_v, read_nums=1, sleep_t=0.1, save_data=False, opm=None,
                      av_mzi=None, av_delta=0, av_sigmas=0):
        """
        Sweep current on phaseshifter. Power reset to 0 on phaseshifter after sweep.
        :param sweep_cs: Array of current values to set.
        :param sweep_v: Constant voltage value.
        :param read_nums: Number of measurements per set current.
        :param sleep_t: Sleep time between two measurements.
        :param save_data: If True, save calibration results and update interpolation data
        :param opm: Optical Power Meter. If none is supplied, optical power will not be recorded
        :param av_mzi: The MZI whose sigma phase is to be averaged over.
        :param av_delta: The delta phase for the av_MZI.
        :param av_sigmas: The sigmas for the av_mzi to be averaged over

        :return: (len(sweep_cs), read_nums, 6) array, where the last layer stores
        [sweep_c, measured_c, measured_v, electric_power, resistance, optical_power] for each measurement of each
        current value.
        """
        channel = self.channel

        optical = True
        if opm is None:
            warnings.warn("Optical power meter not connected, will not record optical power")
            optical = False

        # averaging over previous sigma
        if av_mzi is not None:
            av_mzi_nums = read_nums
            read_nums = 1
        # Result array
        c_results = np.zeros((len(sweep_cs), read_nums, 6))
        try:
            # Set voltage
            self.power_supply.set_voltage(channel, sweep_v)
            print(f'Sweep Channel={channel}, phaseshifter {self.label}, set {sweep_v}V')


            # for i_c, sweep_c in enumerate(sweep_cs):
            for i_c in progressbar(range(len(sweep_cs)), prefix=rf'Sweeping phaseshifter{self.label}'):
                sweep_c = sweep_cs[i_c]

                self.power_supply.set_current(channel, sweep_c)

                #TODO: find the best sleep time.
                sleep(1)

                for i_meas in range(read_nums):
                    v, c = self.power_supply.read_channel(channel)
                    electric_power = v * c  # units mW

                    if c == 0:
                        resistance = np.nan
                    else:
                        resistance = v / c * 1000  # units Ohms

                    if optical:
                        if av_mzi is None:
                            optical_power = opm.read() * 1000  # units mW
                        else:
                            s_results = av_mzi.sweep_sigma(delta=av_delta, sweep_sigmas=av_sigmas, sweep_v=sweep_v,
                                                           read_nums=av_mzi_nums, sleep_t=sleep_t, save_data=False, opm=opm)

                            optical_power = np.mean(s_results[:, :, -1])  # averaged optical power
                    else:
                        optical_power = 0
                    c_results[i_c, i_meas, :] = np.array([sweep_c, c, v, electric_power, resistance, optical_power])
                    sleep(sleep_t)


            self.zero()
            rest_t = 2
            print(f'Sweep complete, zero channel, sleep {rest_t}s')
            sleep(rest_t)  # allow heat to dissipate after sweep

            if save_data:
                np.save(self.sweep_file, c_results)

                # update interpolation data
                mean_results = np.mean(c_results, axis=1)
                sweep_cs = mean_results[:, 0]
                electric_power = mean_results[:, 3]

                self.interp_data = (sweep_cs, electric_power)

        # If sweep stops due to any exception, zero all power supply channels to ensure safety.
        except KeyboardInterrupt:
            print("Shutdown requested... zero all power supply channels")
            self.power_supply.set_voltage(channel, 0)
            self.power_supply.set_current(channel, 0)
            self.power_supply.zero_all()
            raise

        except Exception as e:
            print('Ran into exception, zero all power supply channels...')
            self.power_supply.zero_all()
            raise

        return c_results

    def read_electric_power(self):
        """Return the electric power applied to the phaseshifter. Unit: mW"""
        # v, c = self.power_supply.read_channel(self.channel)
        v, c = self.read_channel()

        return v*c

    def set_current(self, c, voltage_lim=15):
        self.power_supply.set_channel(self.channel, v=voltage_lim, c=c)

    def read_channel(self):
        v, c = self.power_supply.read_channel(self.channel)

        return v, c

    def find_extreme(self, guess_c, opm, read_num=10, min=True, voltage_lim=15, method_args=None):
        """
        Find the extremal point of optical power and corresponding current value
        :param guess_c: Guess current value
        :param opm: optical power meter
        :param read_num: Number of optical power meter readings to take average over
        :param min: If True, find minimum, else find maximum
        :param method_args: Args to pass to scipy.optimize.minimize
        :return: OptimizeResult. Important attributes are: 'x' the solution array, 'fun' value of the optical power,
        'success' a Boolean flag indicating if the optimizer exited successfully and 'message' which describes the
        cause of the termination.
        """
        from scipy.optimize import minimize
        from scipy.optimize import Bounds

        if method_args is None:
            bnds = Bounds(0, 65)
            method_args = {'method': 'Nelder-Mead', 'bounds': bnds, 'options': {'xatol': 0.005}}

        def probe(c):
            c = c[0]
            self.set_current(c, voltage_lim=voltage_lim)
            sleep(1)

            ops = np.zeros(read_num)
            for i in range(read_num):
                ops[i] = opm.read() * 1000
                sleep(0.1)

            op = np.mean(ops)
            print(rf'Set {c}mA, read {op}mW')

            if min:
                return op
            else:
                return -op

        try:
            res = minimize(probe, np.atleast_1d(guess_c), **method_args)
        except KeyboardInterrupt:
            print("Shutdown requested... zero all power supply channels")
            self.set_current(0, 0)
            self.power_supply.zero_all()
            raise
        except Exception as e:
            print('Ran into exception, zero all power supply channels...')
            self.power_supply.zero_all()
            raise

        return res

class ExPhaseshifter(Phaseshifter):
    """External phaseshifter enable setting of phase even if not in an MZI"""

    def __init__(self, mesh_name: str, label: tuple, power_supply=None, channel=None):

        super().__init__(mesh_name, label, power_supply, channel=channel)

        self._params = {}
        self.file = DFUtils.create_filename(config.home_dir + rf'\{mesh_name}\ps_params\{self.label}.json')

        if os.path.isfile(self.file):
            self.read_params()
        else:
            self._params = {
                'k': None,  # heat-phase proportionality, i.e. theta = k x + b, unit=rad / mW
                'b': 0,  # unit: rad
            }

    @property
    def params(self):
        return self._params.copy()

    def read_params(self, file=None):
        """Read params from data file. """
        if file is None:
            file = self.file
        with open(file, 'r') as f:
            params = json.load(f)
        self._params.update(params)

    def save_params(self, file=None):
        if file is None:
            file = self.file
        with open(file, 'w') as f:
            json.dump(self._params, f)

    def update_params(self, updates, save=False):
        self._params.update(updates)
        if save:
            self.save_params()

    def set_phase(self, theta, voltage_lim):
        try:
            params = self.params
            if params['k'] is None:
                raise Exception(rf'Phaseshifter {self.label} not calibrated. ')

            theta = theta - params['b']
            while theta < 0:
                theta = theta + 2 * np.pi
            while theta > 2 * np.pi:
                theta = theta - 2 * np.pi

            x = theta / params['k']
            self.set_power(x, voltage_lim=voltage_lim)

        except Exception as e:
            print('Ran into exception, zero all power supply channels...')
            self.power_supply.zero_all()
            raise


class MZI(Component):
    def __init__(self, mesh_name: str, label: tuple, power_supply=None, channels=(None, None)):
        super().__init__(mesh_name, label, power_supply)

        (i, j) = label
        (up_ch, low_ch) = channels

        self._params = {}
        self.file = DFUtils.create_filename(config.home_dir + rf'\{mesh_name}\mzi_params\{self.label}.json')

        if os.path.isfile(self.file):
            self.read_params()
        else:
            self._params = {
                'alpha': None,  # BS error of first directional coupler
                'beta': None,  # BS error of second directional coupler
                'k1': None,  # heat-phase proportionality of up_ps, i.e. theta1 = k1 x1 + b1, unit=rad / mW
                'k2': None,  # heat-phase proportionality of low_ps, i.e. theta2 = k2 x2 + b2
                'b1-b2': 0,  # unit: rad
                'b1+b2': 0,  # Start off with zero, so it can still be possible to set delta partially when b1+b2 is
                # unknown.
            }

        self._up_ps = Phaseshifter(mesh_name, (i, j), power_supply, channel=up_ch)  # upper phaseshifter
        self._low_ps = Phaseshifter(mesh_name, (i + 1, j), power_supply, channel=low_ch)  # lower phaseshifter

    @property
    def params(self):
        return self._params.copy()

    @property
    def calibrated(self):
        if self._params['k1'] is None or self._params['k2'] is None:
            return False
        else:
            return True

    @property
    def k1(self):
        return self._params['k1']

    @property
    def k2(self):
        return self._params['k2']

    @property
    def b1(self):
        return 0.5 * self._params['b1+b2'] + 0.5 * self._params['b1-b2']

    @property
    def b2(self):
        return 0.5 * self._params['b1+b2'] - 0.5 * self._params['b1-b2']

    @property
    def up_ps(self):
        return self._up_ps

    @property
    def low_ps(self):
        return self._low_ps

    def read_params(self, file=None):
        """Read params from data file. """
        if file is None:
            file = self.file
        with open(file, 'r') as f:
            params = json.load(f)
        self._params.update(params)

    def save_params(self, file=None):
        if file is None:
            file = self.file
        with open(file, 'w') as f:
            json.dump(self._params, f)

    def update_params(self, updates, save=False):
        self._params.update(updates)
        if save:
            self.save_params()

    def sweep_up(self, *args, **kwargs):
        """Sweep current on upper phaseshifter."""
        c_results = self.up_ps.sweep_current(*args, **kwargs)
        return c_results

    def sweep_low(self, *args, **kwargs):
        """ Sweep current on lower phaseshifter."""
        c_results = self.low_ps.sweep_current(*args, **kwargs)
        return c_results

    def set_phase(self, delta, sigma=None, voltage_lim=20):
        try:
            if not self.calibrated:
                raise Exception(rf'MZI {self.label} not calibrated. ')

            b1 = self.b1
            b2 = self.b2

            while delta < -np.pi:
                delta = delta + 2*np.pi
            while delta > np.pi:
                delta = delta - 2*np.pi

            if sigma is None: # doesn't care sigma, so set to zero one phaseshift.
                if b1-delta > b2 + delta:
                    # sigma = b1-delta
                    kx1 = 0.
                    kx2 = b1 - 2 * delta - b2
                else:
                    # sigma = b2 + delta
                    kx1 = 2 * delta + b2 - b1
                    kx2 = 0.
            else:
                while sigma < 0:
                    sigma = sigma + 2*np.pi
                while sigma > 2*np.pi:
                    sigma = sigma - 2*np.pi

                kx1 = sigma + delta - b1
                kx2 = sigma - delta - b2

            # zero if approximately zero
            if np.abs(kx1) < 1e-4:
                kx1 = 0
            if np.abs(kx2) < 1e-4:
                kx2 = 0

            while kx1 < 0:
                kx1 = kx1 + 2 * np.pi
            while kx1 > 2 * np.pi:
                kx1 = kx1 - 2 * np.pi
            while kx2 < 0:
                kx2 = kx2 + 2 * np.pi
            while kx2 > 2 * np.pi:
                kx2 = kx2 - 2 * np.pi

            x1 = kx1 / self.k1
            x2 = kx2 / self.k2
            self.up_ps.set_power(x1, voltage_lim)
            self.low_ps.set_power(x2, voltage_lim)

        except Exception as e:
            print('Ran into exception, zero all power supply channels...')
            self.power_supply.zero_all()
            raise

    def sweep_sigma(self, delta, sweep_sigmas, sweep_v, read_nums, sleep_t=0.1, save_data=False, opm=None):
        """
        Sweep Sigma of the MZI. After sweep, MZI reset to delta and sigma=0
        :param delta: Delta phase, kept constant during and after sweep.
        :param sweep_sigmas: The array of relative sigma phase values to sweep. The sigma values are the
        sum(sweep_sigmas, max(b1, b2)).
        :param sweep_v: Maximum voltage for each phaseshifter during the sweep
        :param read_nums: Number of measurements per sigma swept.
        :param sleep_t: Time between measurement
        :param save_data: If True, save raw sigma sweep data.
        :param opm: Optical power meter. Will not record optical power if None.
        :return:
        """
        try:
            if not self.calibrated:
                raise Exception(rf'MZI {self.label} not calibrated. ')

            optical = True
            if opm is None:
                warnings.warn("Optical power meter not connected, will not record optical power")
                optical = False

            # construct sigmas array by adding max(b1,b2). This reduces the power applied to phaseshifters.
            b1 = self.b1
            b2 = self.b2
            sweep_sigmas = np.atleast_1d(sweep_sigmas)
            sweep_sigmas = sweep_sigmas.astype(float)
            sweep_sigmas += max(b1, b2)

            # Result array
            s_results = np.zeros((len(sweep_sigmas), read_nums, 5))
            # for i_s, sweep_sigma in enumerate(sweep_sigmas):
            for i_s in progressbar(range(len(sweep_sigmas)), prefix=rf'Sweeping sigma on MZI{self.label}'):
                sweep_sigma = sweep_sigmas[i_s]

                self.set_phase(delta=delta, sigma=sweep_sigma, voltage_lim=sweep_v)
                sleep(0.5)

                for i_meas in range(read_nums):
                    x1 = self.up_ps.read_electric_power()
                    x2 = self.low_ps.read_electric_power()

                    kx = self.k1 * x1 + self.k2 * x2

                    if optical:
                        optical_power = opm.read() * 1000  # units mW
                    else:
                        optical_power = 0

                    s_results[i_s, i_meas, :] = np.array([sweep_sigma, x1, x2, kx, optical_power])
                    sleep(sleep_t)

            print(f'Sweep complete, revert sigma, sleep 2s')
            self.set_phase(delta=delta, sigma=None, voltage_lim=sweep_v)
            sleep(2)

            if save_data:
                np.save(DFUtils.create_filename(config.home_dir + rf'\{self.mesh}\sigma_sweep\{self.label}.npy'), s_results)

        # If sweep stops due to any exception, zero all power supply channels to ensure safety.
        except KeyboardInterrupt:
            print("Shutdown requested... zero all power supply channels")
            self.zero()
            self.power_supply.zero_all()
            raise

        except Exception as e:
            print('Ran into exception, zero all power supply channels...')
            self.power_supply.zero_all()
            raise

        return s_results

    def zero(self):
        """Remove all electric power on this MZI"""
        self.up_ps.zero()
        self.low_ps.zero()
