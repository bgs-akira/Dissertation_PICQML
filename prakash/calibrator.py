import warnings

import numpy as np
from scipy.optimize import curve_fit  # CTRL-C crashes Python instead of raising KeyBoardInterrupt when scipy is loaded.
from time import sleep
from abc import ABC, abstractmethod

import prakash.config as config
from prakash.utils import DFUtils
from prakash.driver.optical_switch import Hand
#TODO: Save calibration in some archive file, in case one calibration messed up.
# Also maybe enable calibration from data? Bottom line is some safety mechanism when a certain calibration messed up.

class Calibrator(ABC):

    @abstractmethod
    def __init__(self, mesh, opm, osw=None):

        self.mesh = mesh

        if osw is None:
            self.osw = Hand()
        else:
            self.osw = osw


    @staticmethod
    def construct_sweep_cs(min_c, max_c, data_points, safe_c_step=5.):
        sweep_cs = np.sqrt(np.linspace(min_c ** 2, max_c ** 2, num=data_points))
        while np.any(np.diff(sweep_cs) >= safe_c_step):
            c_steps = np.diff(sweep_cs)
            i = np.argmax(c_steps)
            max_step = c_steps[i]
            sweep_cs = np.insert(sweep_cs, i + 1,
                                 np.linspace(sweep_cs[i], sweep_cs[i + 1], 2 + int(max_step // safe_c_step))[1:-1])

        return sweep_cs

    @abstractmethod
    def calibrate_delta(self, mzi_label):
        pass


class FitCalibrator(Calibrator):

    def __init__(self, mesh, opm, osw=None):
        """
        :param mesh: Mesh object
        :param opm: Optical Power Meter
        :param osw: Optical Switch
        """

        super().__init__(mesh, opm, osw)

        sweep_v = 20
        read_nums = 10
        sleep_t = 0.05
        sweep_cs = self.construct_sweep_cs(0, 65, 300, 0.5)  # default sweep current array
        sweep_sigmas = np.linspace(0, 2*np.pi, num=40)  # default sweep sigma array for calibrating sigma
        av_sigmas = np.linspace(0, 2*np.pi, num=20)  # default sigma array for averaging sigma to trace out leaked light

        # parameters for sweeping phaseshifters
        self.ps_sweep_params = {
            'sweep_cs': sweep_cs, 'sweep_v': sweep_v, 'read_nums': read_nums,
            'sleep_t': sleep_t, 'opm': opm, 'av_delta': 0, 'av_sigmas': av_sigmas, 'save_data': True,
        }

        # parameters for sweeping sigmas
        self.sigma_sweep_params = {
            'sweep_sigmas': sweep_sigmas, 'sweep_v': sweep_v, 'read_nums': read_nums,
            'sleep_t': sleep_t, 'opm': opm, 'save_data': True
        }

        # TODO: the calibrator doesn't need to be so specific with these. Can write a separate function calibrate_all() which deals with the osw and channels etc.
        # Default optical switch mapping is mode 0 -> switch channel 0 ... mode 9 -> switch channel 9
        osw_mapping = {}
        for mode in range(10):
            osw_mapping[mode]=mode
        self._osw_mapping = osw_mapping

        self.output_modes = {4: 0, 5: 1, 6: 3, 7: 5, 8: 7}
        self.reverse_output_modes = {0: 1, 1: 3, 2: 5, 3: 7}

    @staticmethod
    def pp_fit_func(x, a, c, k, b):
        return a + c * np.sin(k * x + b)

    @staticmethod
    def fit_ps_sweeps(c_results, p0=None):
        mean_results = np.mean(c_results, axis=1)
        if p0 is None:
            p0 = [0.5, 0.5, 2 * np.pi / 700, np.pi / 2]
        try:
            popt, _ = curve_fit(Calibrator.pp_fit_func, mean_results[:, 3], mean_results[:, 5] / np.max(mean_results[:, 5]),
                                p0=p0)  # Fit normalised optical power vs electric power
            return popt
        except RuntimeError as error:
            warnings.warn(f'Error: {error}. popt will be p0.')
            return p0

    @staticmethod
    def fit_sigma_sweeps(s_results, kxs_modified, p0=None):
        mean_results = np.mean(s_results, axis=1)
        if p0 is None:
            p0 = [0.5, 0.5, 1., np.pi / 2]
        try:
            popt, _ = curve_fit(Calibrator.pp_fit_func, 0.5 * kxs_modified, mean_results[:, 4] / np.max(mean_results[:, 4]),
                                p0=p0)
            return popt
        except RuntimeError as error:
            warnings.warn(f'Error: {error}. popt will be p0.')
            return p0

    @property
    def osw_mapping(self):
        return self._osw_mapping.copy()

    @osw_mapping.setter
    def osw_mapping(self, mapping):
        self._osw_mapping = mapping

    #TODO: rewrite the functions to include setting previous diagonals to be cross.
    def calibrate_delta(self, mzi_label, save_param=True, av_mzi_label=None, backup_dir=None, **kwargs):
        targetMZI = self.mesh.mzi(mzi_label)
        if av_mzi_label is None:
            avMZI = None
        else:
            avMZI = self.mesh.mzi(av_mzi_label)

        # Initial estimate
        params = targetMZI.params
        if params['k1'] is None:
            p0 = [0.5, 0.5, 2 * np.pi / 700, np.pi / 2]
        else:
            p0 = [0.5, 0.5, params['k1'], np.pi / 2 + params['b1-b2']]

        # Construct sweep parameters
        ps_sweep_params = self.ps_sweep_params.copy()
        ps_sweep_params.update(kwargs)

        # Sweep upper internal phaseshifter
        results1 = targetMZI.sweep_up(av_mzi=avMZI, **ps_sweep_params)
        popt1 = self.fit_ps_sweeps(results1, p0=p0)

        params['k1'] = popt1[2]
        params['b1-b2'] = popt1[3] - np.pi / 2

        print(f'popt1={popt1}, sleep 5s')
        sleep(5)

        # Update initial guess
        p0 = popt1.copy()
        p0[3] = np.pi - p0[3]

        # Sweep low internal phaseshifter
        results2 = targetMZI.sweep_low(av_mzi=avMZI, **ps_sweep_params)

        # Fit to find k2
        popt2 = self.fit_ps_sweeps(results2, p0=p0)

        params['k2'] = popt2[2]

        print(f'popt2={popt2}, sleep 5s')
        sleep(5)

        # Update MZI parameter
        targetMZI.update_params(params, save=save_param)

        if backup_dir is not None:
            (i,j) = mzi_label
            np.save(DFUtils.create_filename(backup_dir + rf'\ps_sweep\{(i,j)}.npy'), results1)
            np.save(backup_dir + rf'\ps_sweep\{(i+1,j)}.npy', results2)

        # There is an error between b1-b2 and b2-b1.
        return popt1, popt2

    def calibrate_sigma(self, mzi_label, save_param=True, reverse=False, **kwargs):
        (i,j) = mzi_label
        targetMZI = self.mesh.mzi(mzi_label)

        # Initial estimate
        params = targetMZI.params
        p0 = [0.5, 0.5, 1., np.pi/2 + 0.5*params['b1+b2']]

        if reverse:
            effBSs = [(i+1, j-1), (i+1, j+1)]
            effPSs = [mzi_label, (i+2, j)]

            upperMZI = self.mesh.mzi((i+2, j))

        else:
            effBSs = [(i - 1, j - 1), (i - 1, j + 1)]
            effPSs = [mzi_label, (i-2, j)]

            upperMZI = self.mesh.mzi((i-2, j))

        # Construct meta MZI.
        self.mesh.set_mzi(effPSs, delta=np.pi/2, sigma=0)
        sleep(0.5)
        self.mesh.set_mzi(effBSs, delta=np.pi/4)
        sleep(0.5)

        # Construct sweep parameters
        sigma_sweep_params = self.sigma_sweep_params.copy()
        sigma_sweep_params.update(kwargs)

        # Sweep Sigma
        # the last layer of s_results are [set_sigma, x1, x2, k1*x1+k2*x2, optical_power]
        # the target mzi will reset back to delta=pi/2 and sigma=0 after sweep.
        s_results = targetMZI.sweep_sigma(delta=np.pi/2, **sigma_sweep_params)
        mean_results = np.mean(s_results, axis=1)
        (sweep_sigmas, x1s, x2s, kxs, optical_power) = tuple(mean_results.T)

        # Modify k1x1 + k2x2
        k1 = targetMZI.k1
        k2 = targetMZI.k2
        b_diff = targetMZI.params['b1-b2']
        kxs_modified = np.zeros_like(kxs)
        for i_d in range(len(kxs)):
            x1 = x1s[i_d]
            x2 = x2s[i_d]
            kx = kxs[i_d]
            if k1 * x1 - k2 * x2 + b_diff < 0:
                if kx < 2 * np.pi:
                    kx = kx + 2 * np.pi
                else:
                    kx = kx - 2 * np.pi
            kxs_modified[i_d] = kx

        popt = self.fit_sigma_sweeps(s_results, kxs_modified, p0=p0)

        # Calculate b1+b2 relative to the main diagonal. popt[3]-pi/2 is relative to the upper MZI, which we need to
        # account for.
        upper_params = upperMZI.params
        params['b1+b2'] = 2 * popt[3] - np.pi + upper_params['b1+b2']

        # Update MZI parameter
        targetMZI.update_params(params, save=save_param)
        print(f'popt={popt}')
        return popt

    def calibrate_theta(self, ps_label, save_param=True):
        """Calibrate external phaseshifter"""
        (i, j) = ps_label
        top = (i==0)  # whether this is external phaseshifter in the first row

        targetPS = self.mesh.phaseshifter(ps_label)

        # Initial estimate
        params = targetPS.params
        if params['k'] is None:
            p0 = [0.5, 0.5, 2 * np.pi / 700, 0]
        else:
            p0 = [0.5, 0.5, params['k'], params['b']]

        # Construct meta MZI
        if top:
            self.mesh.set_mzi((i + 1, j), delta=np.pi / 2, sigma=0)
            self.mesh.set_mzi((i, j - 1), delta=np.pi / 4)
            self.mesh.set_mzi((i, j + 1), delta=np.pi / 4)

            upperMZI = self.mesh.mzi((i+1, j))
        else:
            self.mesh.set_mzi((i - 2, j), delta=np.pi / 2, sigma=0)
            self.mesh.set_mzi((i - 1, j - 1), delta=np.pi / 4)
            self.mesh.set_mzi((i - 1, j + 1), delta=np.pi / 4)

            upperMZI = self.mesh.mzi((i - 2, j))

        # Sweep Sigma
        # the columns of s_results are [set_sigma, x1, x2, k1*x1+k2*x2, optical_power]
        # the mzi will reset back to delta=pi/2 and sigma=0 after sweep.
        c_results = targetPS.sweep_current(**self.ps_sweep_params)
        popt = self.fit_ps_sweeps(c_results, p0=p0)

        # proportionality between phase and heat
        params['k'] = popt[2]

        # Calculate b relative to the main diagonal. popt[3]-pi/2 is relative to the upper MZI, which we need to
        # account for.

        upper_params = upperMZI.params
        params['b'] = popt[3] + upper_params['b1+b2']

        # Update MZI parameter
        targetPS.update_params(params, save=save_param)

        return popt

    def calibrate_main(self, save_param=True):
        try:
            output_mode = self.output_modes[4]
            self.osw.switch(self.osw_mapping[output_mode])

            popts = {}

            components = self.mesh.get_diagonal(k=4)
            for mzi_label in components['mzi']:
                popt1, popt2 = self.calibrate_delta(mzi_label, save_param=save_param)
                popts[mzi_label] = (popt1, popt2)

        except Exception as e:
            print('Ran into exception, zero all power supply channels...')
            self.mesh.zero_all()
            raise

        return popts

    # TODO: Add kwargs to this function as well
    def calibrate_diag(self, k, save_param=True, averaging=False):
        """Calibrate diagonal k. Last layer global phaseshift not calibrated. """
        fitting_results = {'delta': {}, 'sigma': {}, 'theta': {}}

        # Find out whether it is main diagonal, lower diagonal or upper diagonal.
        if k==4:
            popts = self.calibrate_main(save_param=save_param)
            fitting_results['delta'] = popts

            return fitting_results

        elif k in self.output_modes.keys():
            reverse = False
            output_mode = self.output_modes[k]

            prev_diags = list(range(4, k))
            bar_row = 8
            last_layer = 9
            k_prev = k-1

            # Dictionary of components and results
            components = self.mesh.get_diagonal(k)
        elif k in self.reverse_output_modes.keys():
            reverse = True
            output_mode = self.reverse_output_modes[k]

            prev_diags = list(range(k+1, 5))[::-1]
            bar_row = 0
            last_layer = 0
            k_prev = k+1

            # Dictionary of components and results
            components = self.mesh.get_diagonal(k)
            components['mzi'].reverse()
        else:
            raise ValueError(f'Diagonal k={k} not found')

        try:
            self.osw.switch(self.osw_mapping[output_mode])
            sleep(1)

            # Set previous diagonal to be cross
            prev_comp = self.mesh.get_diagonal(k_prev)
            self.mesh.set_mzi(prev_comp['mzi'], delta=0)
            sleep(0.5)

            # Set bottom mzis to be bar
            for k2 in prev_diags:
                self.mesh.set_mzi((bar_row, 2*k2-bar_row), delta=np.pi/2)

            # TODO: Enable averaging over previous Sigma to remove error from leaked light.
            # Calibrate MZIs
            for mzi_label in components['mzi']:
                (i,j) = mzi_label

                # Calibrate delta. Both phaseshifters in MZI reset to 0 power after sweep.
                pp_popt1, pp_popt2 = self.calibrate_delta(mzi_label, save_param=save_param)
                fitting_results['delta'][mzi_label] = (pp_popt1, pp_popt2)

                # Calibrate sigma for MZIs except in the last layer
                if mzi_label[1] == last_layer:
                    self.mesh.set_mzi(mzi_label, delta=0)  # set to be delta=0 to maximise light output at the next delta calibration
                    continue
                else:
                    # Make light go into previous diagonal
                    self.mesh.set_mzi((bar_row, 2*k_prev - bar_row), delta=0)

                    # Calibrate sigma. A meta MZI is constructed and remains after the sweep.
                    # TODO: The sigma calibration is for some reason not working. Don't save parameter! Popt data will be kept.
                    popt = self.calibrate_sigma(mzi_label, save_param=False, reverse=reverse)
                    fitting_results['sigma'][mzi_label] = popt

                    # Deconstruct meta MZI
                    self.mesh.set_mzi(prev_comp['mzi'], delta=0)
                    sleep(0.5)
                    if reverse:
                        self.mesh.set_mzi([mzi_label, (i+1, j-1)], delta=0)
                    else:
                        self.mesh.set_mzi([mzi_label, (i-1, j+1)], delta=0)  # set them to be delta=0 to maximise light output
                    sleep(0.5)

                    # Reset bottom MZI
                    self.mesh.set_mzi((bar_row, 2*k_prev - bar_row), delta=np.pi/2)
                    sleep(0.5)



            # Calibrate external phaseshifter
            for ps_label in components['ps']:
                # Calibrate theta
                popt = self.calibrate_theta(ps_label, save_param=save_param)
                fitting_results['theta'][ps_label] = popt

            # zero all
            self.mesh.zero_all()

        except Exception as e:
            print('Ran into exception, zero all power supply channels...')
            self.mesh.zero_all()
            raise

        return fitting_results


class InterpCalibrator(Calibrator):

    def __init__(self, mesh, opm, osw=None):
        """
        :param mesh: Mesh object
        :param opm: Optical Power Meter
        :param osw: Optical Switch
        """

        super().__init__(mesh, opm, osw)

        # these are the default sweeping parameters.
        sweep_v = 20
        read_nums = 10
        sleep_t = 0.05
        sweep_cs = self.construct_sweep_cs(0, 65, 300, 0.5)  # default sweep current array
        sweep_sigmas = np.linspace(0, 2*np.pi, num=40)  # default sweep sigma array for calibrating sigma
        av_sigmas = np.linspace(0, 2*np.pi, num=20)  # default sigma array for averaging sigma to trace out leaked light

        # parameters for sweeping phaseshifters. As long as av_mzi stays None, then no averaging occurs.
        self.ps_sweep_params = {
            'sweep_cs': sweep_cs, 'sweep_v': sweep_v, 'read_nums': read_nums,
            'sleep_t': sleep_t, 'opm': opm, 'av_delta': 0, 'av_sigmas': av_sigmas, 'save_data': True,
        }

        # parameters for sweeping sigmas
        self.sigma_sweep_params = {
            'sweep_sigmas': sweep_sigmas, 'sweep_v': sweep_v, 'read_nums': read_nums,
            'sleep_t': sleep_t, 'opm': opm, 'save_data': True
        }

    def calibrate_delta(self, mzi_label, save_param=True, av_mzi_label=None, backup_dir=None, **kwargs):
        """
        Function to calibrate delta phase of an MZI
        :param mzi_label: The label of the target MZI
        :param save_param: Whether to save the calibration results.
        :param av_mzi_label: The label of the MZI, whose Sigma phase is to be averaged over. If None, no averaging will take place.
        :param backup_dir: The directory to backup the calibration raw data.
        :param kwargs: Extra arguments to pass to Phaseshifter.sweep_current()

        :return:
        """

        '''Target MZI'''
        targetMZI = self.mesh.mzi(mzi_label)

        '''Averaged MZI'''
        if av_mzi_label is None:
            avMZI = None
        else:
            avMZI = self.mesh.mzi(av_mzi_label)

        '''Sweep parameters'''
        ps_sweep_params = self.ps_sweep_params.copy()
        ps_sweep_params.update(kwargs)

        '''Sweep up'''
        results1 = targetMZI.sweep_up(av_mzi=avMZI, **ps_sweep_params)
        mean1 = np.mean(results1, axis=1)




