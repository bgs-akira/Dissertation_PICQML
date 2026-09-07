import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

import prakash.config as config
from prakash.calibrator import Calibrator
from prakash.components import MZI
from prakash.utils import DFUtils

def plot_ps_sweeps(phaseshifter, mesh_name, c_results=None, plot_dir=None, pp_popt=None, normalized=True, **plotkwargs):
    if plot_dir is None:
        to_save = False
    else:
        to_save = True
        os.makedirs(plot_dir, exist_ok=True)

    if c_results is None:
        try:
            c_results = np.load(config.home_dir + rf'\{mesh_name}\ps_sweep\{phaseshifter}.npy')
        except FileNotFoundError:
            raise ValueError('Not existing calibration file found, please provide results array')

    if c_results.shape[2] == 5:
        # Does not contain optical power measurements
        optical = False
    elif c_results.shape[2] == 6:
        # Does contain optical power measurements
        optical = True
    else:
        raise ValueError('c_results shape not recognized')

    # each row is a different sweep_c, storing [sweep_c, mean_c, mean_v, mean_electric_power, mean_resistance, (mean_optical_power)]

    # Average raw data
    mean_results = np.mean(c_results, axis=1)
    std_results = np.std(c_results, axis=1)
    (sweep_cs, mean_cs, mean_vs, electric_power, resistance, optical_power) = tuple(mean_results.T)

    # normalise power
    max_optical_power = np.max(optical_power)
    ind_max = np.argmax(optical_power)
    min_optical_power = np.min(optical_power)
    ind_min = np.argmin(optical_power)
    if normalized:
        optical_power = optical_power / max_optical_power
        std_results[:, 5] = std_results[:, 5] / max_optical_power

    # Find popt
    if pp_popt is None:
        pp_popt = Calibrator.fit_ps_sweeps(c_results)

    # Plot parameters
    full_power_range = np.linspace(0, 800, num=800)
    plot_params = {'num': f'ps{phaseshifter}', 'figsize': (16, 8), 'layout': 'constrained'}
    plot_params.update(plotkwargs)

    fig, axs = plt.subplots(2, 2, **plot_params)
    axs = axs.flatten()

    # VI plot
    # plt.figure(f'{phaseshifter} vi plot')
    ax = axs[0]
    ax.errorbar(mean_cs, mean_vs, xerr=std_results[:, 1], yerr=std_results[:, 2],
                fmt='.', linestyle='None', label='Data')
    ax.set_xlabel('Measured Current (mA)')
    ax.set_ylabel('Measured Voltage (V)')
    ax.set_title('V vs I')
    ax.legend()

    # PI plot
    ax = axs[1]
    ax.errorbar(mean_cs, electric_power, xerr=std_results[:, 1], yerr=std_results[:, 3],
                fmt='.', linestyle='None', label='Data')
    ax.set_xlabel('Set Current (mA)')
    ax.set_ylabel('Measured Electric Power (mW)')
    ax.set_title('EP vs I')
    ax.legend()

    if optical:
        # PP plot
        ax = axs[2]
        ax.errorbar(electric_power, optical_power, xerr=std_results[:, 3], yerr=std_results[:, 5],
                    fmt='.', linestyle='None', label='Data', alpha=0.5)
        ax.plot(full_power_range, Calibrator.pp_fit_func(full_power_range, *pp_popt), label='Fit')
        ax.text(electric_power[ind_max] - 20, optical_power[ind_max], f'{max_optical_power:.3g}mW')
        ax.text(electric_power[ind_min] + 50, optical_power[ind_min], f'{min_optical_power:.3g}mW')
        ax.set_xlabel('Measured Electric power (mW)')
        ax.set_ylabel('Normalized Optical power (mW)')
        ax.set_title(f'ER={max_optical_power / min_optical_power:.3g}')
        ax.set_ylim(bottom=-0.1)
        ax.legend()

        # PI plot
        ax = axs[3]
        ax.errorbar(mean_cs, optical_power, xerr=std_results[:, 1], yerr=std_results[:, 5],
                    fmt='.', linestyle='None', label='Data')
        ax.set_xlabel('Set current (mA)')
        ax.set_ylabel('Normalized Optical power')
        ax.set_title('OP vs I')
        ax.legend()

    if to_save:
        plt.savefig(DFUtils.create_filename(plot_dir + rf'\{phaseshifter}_cali_plots.pdf'))

    return fig, axs


def plot_mzi_sweeps(mzi_label, mesh_name, c_results1=None, c_results2=None, plot_dir=None, popt1=None, popt2=None,
                    normalized=True, **plotkwargs):
    if plot_dir is None:
        to_save = False
    else:
        to_save = True
        os.makedirs(plot_dir, exist_ok=True)

    (i, j) = mzi_label

    if c_results1 is None:
        try:
            c_results1 = np.load(config.home_dir + rf'\{mesh_name}\ps_sweep\{(i, j)}.npy')
        except FileNotFoundError:
            raise ValueError('Not existing calibration file found, please provide results array')

    if c_results2 is None:
        try:
            c_results2 = np.load(config.home_dir + rf'\{mesh_name}\ps_sweep\{(i + 1, j)}.npy')
        except FileNotFoundError:
            raise ValueError('Not existing calibration file found, please provide results array')

    mean1 = np.mean(c_results1, axis=1)
    std1 = np.std(c_results1, axis=1)
    (_, _, _, ep1, _, op1) = tuple(mean1.T)
    if popt1 is None:
        popt1 = Calibrator.fit_ps_sweeps(c_results1)

    mean2 = np.mean(c_results2, axis=1)
    std2 = np.std(c_results2, axis=1)
    (_, _, _, ep2, _, op2) = tuple(mean2.T)
    if popt2 is None:
        popt2 = Calibrator.fit_ps_sweeps(c_results2)

    # normalise
    if normalized:
        max_op1 = np.max(op1)
        op1 = op1 / max_op1
        std1[:, 5] = std1[:, 5] / max_op1

        max_op2 = np.max(op2)
        op2 = op2 / max_op2
        std2[:, 5] = std2[:, 5] / max_op2

    full_power_range = np.linspace(0, 800, num=40)

    # Plot parameters
    plot_params = {'num': f'mzi{mzi_label}', 'figsize': (8, 4)}
    plot_params.update(plotkwargs)
    fig, axs = plt.subplots(2, 1, **plot_params)

    # PP plot
    ax = axs[0]
    ax.errorbar(ep1, op1, xerr=std1[:, 3], yerr=std1[:, 5], alpha=0.5,
                fmt='.', linestyle='None', label=f'Data {(i, j)}')
    ax.plot(full_power_range, Calibrator.pp_fit_func(full_power_range, *popt1) * np.max(op1), label=f'Fit {(i, j)}')
    halfpi_pos = (np.pi / 2 - popt1[3]) / popt1[2]
    ax.axvline(halfpi_pos, color='black', linestyle='--')
    ax.text(halfpi_pos + 20, 0, f'b1-b2={popt1[3] - np.pi / 2:.3f}')
    ax.set_ylim(bottom=-0.1)

    ax.set_ylabel('Normalized Optical power (mW)')
    ax.legend()


    ax = axs[1]
    ax.errorbar(ep2, op2, xerr=std2[:, 3], yerr=std2[:, 5],  alpha=0.5,
                fmt='.', linestyle='None', label=f'Data {(i + 1, j)}')
    ax.plot(full_power_range, Calibrator.pp_fit_func(full_power_range, *popt2) * np.max(op2), label=f'Fit {(i + 1, j)}')
    halfpi_pos = (np.pi / 2 - popt2[3]) / popt2[2]
    ax.axvline(halfpi_pos, color='black', linestyle='--')
    ax.text(halfpi_pos - 20, 0.2, f'b2-b1={popt2[3] - np.pi / 2:.3f}')

    ax.set_ylim(bottom=-0.1)

    ax.set_xlabel('Measured Electric power (mW)')
    ax.set_ylabel('Normalized Optical power (mW)')
    ax.legend()

    if to_save:
        plt.savefig(DFUtils.create_filename(plot_dir + rf'\mzi_{mzi_label}.pdf'))

    return fig, axs


def plot_sigma_sweeps(mzi_label, mesh_name, s_results=None, plot_dir=None, popt=None, normalized=True,
                      **plotkwargs):
    if plot_dir is None:
        to_save = False
    else:
        to_save = True
        os.makedirs(plot_dir, exist_ok=True)

    if s_results is None:
        try:
            s_results = np.load(config.home_dir + rf'\{mesh_name}\sigma_sweep\{mzi_label}.npy')
        except FileNotFoundError:
            raise ValueError('Not existing calibration file found, please provide results array')

    # each row is a different sweep_c, storing [sweep_sigma, x1, x2, k1x1+k2x2, optical_power]
    mean_results = np.mean(s_results, axis=1)
    std_results = np.std(s_results, axis=1)
    (sweep_sigmas, x1s, x2s, kxs, optical_power) = tuple(mean_results.T)

    # find k1 and k2
    targetMZI = MZI(mesh_name, mzi_label)
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

    if popt is None:
        popt = Calibrator.fit_sigma_sweeps(s_results, kxs_modified)

    max_optical_power = np.max(optical_power)
    if normalized:
        optical_power = optical_power / max_optical_power
        std_results[:, -1] = std_results[:, -1] / max_optical_power

    phase_range = np.linspace(0, 2 * np.pi, num=40)

    # plot parameters
    plot_params = {'num': f'sigma for mzi{mzi_label}', 'figsize': (16, 8), 'layout': 'constrained'}
    plot_params.update(plotkwargs)
    fig, axs = plt.subplots(2, 2, **plot_params)
    axs = axs.flatten()

    # plot optical power against kxs
    ax = axs[0]
    ax.errorbar(0.5*kxs_modified / np.pi, optical_power, xerr=std_results[:, 3]/np.pi, yerr=std_results[:, 4],
                fmt='x', linestyle='None', label='Data')
    ax.plot(phase_range / np.pi, Calibrator.pp_fit_func(phase_range, *popt), label='Fit')
    ax.text(0, 0.5, rf'relative $b_1+b_2=${popt[3] / np.pi - 0.5:.3f}$\pi$')
    ax.set_ylabel('Normalized Optical power')
    ax.set_xlabel(r'Modified $\frac{k_1x_1+k_2x_2}{2} / \pi$)')
    ax.set_ylim(bottom=-0.1)
    ax.legend()

    # plot optical power against set sigma
    ax = axs[1]
    ax.errorbar(sweep_sigmas / np.pi, optical_power, yerr=std_results[:, 4],
                fmt='x', linestyle='None', label='Data')
    ax.plot(phase_range / np.pi, Calibrator.pp_fit_func(phase_range, *popt), label='Fit')
    ax.set_xlabel(r'Set $\Sigma / \pi$)')
    ax.set_ylabel('Normalized Optical power')
    ax.set_ylim(bottom=-0.1)
    ax.legend()

    # plot k1x1 and k2x2 against set sigma
    ax = axs[2]
    ax.errorbar(sweep_sigmas / np.pi, k1 * x1s / np.pi, yerr=std_results[:, 1] * k1 / np.pi, fmt='x', linestyle='None',
                label=r'$k_1x_1$')
    ax.errorbar(sweep_sigmas / np.pi, k2 * x2s / np.pi, yerr=std_results[:, 2] * k2 / np.pi, fmt='x', linestyle='None',
                label=r'$k_2x_2$')
    ax.plot(sweep_sigmas / np.pi, 0.5 * (k1 * x1s - k2 * x2s + b_diff) / np.pi, label=r'$\delta')
    ax.axhline(0.5, linestyle='--', alpha=0.5)
    ax.axhline(-0.5, linestyle='--', alpha=0.5)
    ax.set_xlabel(r'Set $\Sigma / \pi$)')
    ax.set_ylabel('Measured phase $/ \pi$)')
    ax.legend()

    # Plot modified k1x1 + k2x2
    ax = axs[3]
    ax.errorbar(sweep_sigmas / np.pi, kxs / 2 / np.pi, yerr=std_results[:, 3] / np.pi, fmt='x',
                linestyle='None', label='measured')
    ax.errorbar(sweep_sigmas / np.pi, kxs_modified / 2 / np.pi, yerr=std_results[:, 3] / np.pi, fmt='x',
                linestyle='None', label='modified')
    ax.plot(sweep_sigmas / np.pi, 0.5 * (k1 * x1s - k2 * x2s + b_diff) / np.pi, label='delta')
    ax.axhline(0.5, linestyle='--', alpha=0.5)
    ax.axhline(-0.5, linestyle='--', alpha=0.5)
    ax.set_xlabel('Set sigma/pi')
    ax.set_ylabel(r'$\frac{k_1x_1+k_2x_2}{2} / \pi$)')
    ax.legend()

    if to_save:
        plt.savefig(DFUtils.create_filename(plot_dir + rf'\mzi_{mzi_label}_sigma_cali.pdf'))

    return fig, axs


if __name__ == "__main__":
    plot_dir = r'C:\Users\PhotonsLocalAdmin\Documents\prakash_one_control\Results\compare_av_sigma\2023-11-16(17-38-04.942854)\(1,9)_without'

    plot_ps_sweeps((1, 9), 'prakash_one', plot_dir=plot_dir)

    plot_ps_sweeps((2, 9), 'prakash_one', plot_dir=plot_dir)
