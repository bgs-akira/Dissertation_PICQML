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

mesh_name = 'prakash_one'
mzi_label = (2,6)
i,j = mzi_label

# plot_dir = r'C:\Users\PhotonsLocalAdmin\Documents\prakash_one_control\Results\test_delta_sweep\mzi_(0, 8)_2024-03-06(19-23-17.512744)'
plot_dir =config.home_dir + rf'\..\Results\test_delta_sweep\mzi_{mzi_label}_{config.time_stamp}'

'''Connect devices '''
power_supply = XPOWu(['COM15', 'COM16', 'COM17'])
# Find resource by pyvisa.ResourceManager().list_resources()
opm = PM100D('USB0::0x1313::0x8078::P0022518::INSTR')  # optical power meter

pMesh = Mesh(name=mesh_name, power_supply=power_supply)

'''Sweep parameters'''
sweep_cs = Calibrator.construct_sweep_cs(min_c=0, max_c=65, data_points=300, safe_c_step=2)
read_nums = 1
sleep_t = 0.1
sweep_v = 15
sleep_between_ps = 5
test_params = {
    'sweep_cs': sweep_cs, 'sweep_v': sweep_v, 'read_nums': read_nums, 'sleep_t': sleep_t,
    'sleep_between_ps': sleep_between_ps
}

sweep_params = {
    'sweep_cs': sweep_cs,
    'sweep_v': sweep_v,
    'read_nums': read_nums,
    'sleep_t': sleep_t,
    'opm': opm,
}

with open(DFUtils.create_filename(plot_dir + rf'\test_params.p'), 'wb') as fp:
    pickle.dump(test_params, fp)

targetMZI = pMesh.mzi(mzi_label)

'''Sweep upper PS'''
results1 = targetMZI.sweep_up(**sweep_params)
np.save(plot_dir + rf'\{targetMZI.up_ps.label}_c_results.npy', results1)
# results1 = np.load(plot_dir + rf'\({i}, {j})_c_results.npy')

mean_results1 = np.mean(results1, axis=1)

p0 = [0.5, 0.5, 2 * np.pi / 700, np.pi / 2]
popt1 = Calibrator.fit_ps_sweeps(results1, p0=p0)

fig1, axs1 = plot_ps_sweeps((i,j), mesh_name, c_results=results1,
                            pp_popt=popt1, plot_dir=plot_dir)


'''Sleep'''
for i_s in progressbar(range(sleep_between_ps), prefix=f'Sleep {sleep_between_ps}s'):
    sleep(1)

print(rf'Read up_ps channel: {power_supply.read_channel(targetMZI.up_ps.channel)}')
print(rf'Read low_ps channel: {power_supply.read_channel(targetMZI.low_ps.channel)}')


'''Sweep lower PS'''
results2 = targetMZI.sweep_low(**sweep_params)
np.save(plot_dir + rf'\{targetMZI.low_ps.label}_c_results.npy', results2)
# results2 = np.load(plot_dir + rf'\({i+1}, {j})_c_results.npy')
mean_results2 = np.mean(results2, axis=1)

p0[-1] = np.pi - popt1[-1]
popt2 = Calibrator.fit_ps_sweeps(results2, p0=p0)

fig2, axs2 = plot_ps_sweeps((i+1,j), mesh_name, c_results=results2,
                            pp_popt=popt2, plot_dir=plot_dir)


'''Plot both'''
fig3, axs3 = plot_mzi_sweeps(mzi_label, mesh_name, c_results1=results1, c_results2=results2,
                             popt1=popt1, popt2=popt2, plot_dir=plot_dir,
                             figsize=(10, 10))


'''Plot both onto same plot'''
fig4, ax4 = plt.subplots(figsize=(10,5), layout='constrained')

ax4.plot(mean_results1[:, 3], mean_results1[:, 5], '.', ls='None', label='up', alpha=0.5)
ax4.plot(-mean_results2[:, 3], mean_results2[:, 5], '.', ls='None', label='low', alpha=0.5)

xs = np.linspace(0, 800, 500)
# ax4.plot(xs, np.max(mean_results1[:, 5]) * Calibrator.pp_fit_func(xs, *popt1), label='fit up', alpha=1.)
# ax4.plot(-xs, np.max(mean_results2[:, 5]) * Calibrator.pp_fit_func(-xs, *popt2), label='fit low', alpha=1.)

ax4.set_ylabel('Optical power/mW')
ax4.set_xlabel('Electric power/mW')

ax4.legend()

fig4.savefig(plot_dir + rf'\mzi_{mzi_label}_combined.pdf')


'''Fit include second order'''
from scipy.optimize import curve_fit
def pp_fit_func2(x, a, c, k, q, b):
    return a + c * np.sin(k*x + q*x**2 + b)

qp0= [0.5, 0.5, 2 * np.pi / 700, 0., np.pi / 2]
qopt1, qcov1 = curve_fit(pp_fit_func2, mean_results1[:, 3], mean_results1[:, 5] / np.max(mean_results1[:, 5]),
                         p0=qp0)

qp0[-1] = np.pi - qopt1[-1]
qopt2, qcov2 = curve_fit(pp_fit_func2, mean_results2[:, 3], mean_results2[:, 5] / np.max(mean_results2[:, 5]),
                         p0=qp0)


fig5, axs5 = plt.subplots(2,1, sharex='all', sharey='all', layout='constrained', figsize=(10, 8))

i_d = -1
for ax, mean_results, qopt in zip(axs5, [mean_results1, mean_results2], [qopt1, qopt2]):
    i_d += 1
    ax.plot(mean_results[:, 3], mean_results[:, 5] / np.max(mean_results[:, 5]), '.', alpha=0.5, label=f'Data {(i+i_d,j)}')
    ax.plot(xs, pp_fit_func2(xs, *qopt), alpha=1., label='Quadratic Fit')

    halfpi_pos = (np.pi / 2 - qopt[-1]) / qopt[2]
    ax.axvline(halfpi_pos, color='black', linestyle='--')
    ax.text(halfpi_pos + 20, 0, f'b1-b2={qopt[-1] - np.pi / 2:.3f}')

    ax.set_ylim(bottom=-0.1)
    ax.set_ylabel('Normalized Optical power (mW)')
    ax.legend()

ax.set_xlabel('Measured Electric power (mW)')
fig5.savefig(plot_dir + rf'\mzi_{mzi_label}_quadratic_fit.pdf')


'''Fit only a fraction'''
fraction = 1/2
num_datapoints = int(fraction * len(results1))
popt11 = Calibrator.fit_ps_sweeps(results1[:num_datapoints, :, :])
p0 = [0.5, 0.5, 2 * np.pi / 700, np.pi - popt11[-1]]
popt22 = Calibrator.fit_ps_sweeps(results2[:num_datapoints, :, :], p0=p0)


fig6, axs6 = plot_mzi_sweeps(mzi_label, mesh_name, c_results1=results1[:num_datapoints,:,:], c_results2=results2[:num_datapoints, :, :],
                             popt1=popt11, popt2=popt22, plot_dir=None, normalized=False,
                             figsize=(10, 10), num=f'fit first {num_datapoints} points')

axs6[0].plot(mean_results1[num_datapoints:, 3], mean_results1[num_datapoints:, 5], '.', color='green', alpha=0.5)
axs6[1].plot(mean_results2[num_datapoints:, 3], mean_results2[num_datapoints:, 5], '.', color='green', alpha=0.5)

fig6.savefig(plot_dir + rf'\mzi_{mzi_label}_fit_first_{num_datapoints}points.pdf')


'''Fit ep against v and c'''
(set_c1, c1, v1, ep1, _, op1) = tuple(mean_results1.T)
fig7, axs7 = plt.subplots(2, 1, figsize=(10, 8), layout='constrained')
ax = axs7[0]
ax.plot(c1, ep1, '.', alpha=0.5, label='data')
ax.set_xlabel('Current (mA)')
ax.set_ylabel('Electric Power (mW)')

pfit1 = np.polyfit(c1 ,ep1, deg=2)
ax.plot(c1, pfit1[0] * c1 ** 2 + pfit1[1] * c1 + pfit1[2], label='deg=2 fit')

pfit1 = np.polyfit(c1 ,ep1, deg=3)
ax.plot(c1, pfit1[0] * c1 ** 3 + pfit1[1] * c1 ** 2 + pfit1[2] * c1 + pfit1[3], label='deg=3 fit')

pfit1 = np.polyfit(c1 ,ep1, deg=4)
ax.plot(c1, pfit1[0] * c1 ** 4 + pfit1[1] * c1 ** 3 + pfit1[2] * c1 ** 2 + pfit1[3] * c1 + pfit1[4], label='deg=4 fit')

ax.legend()

ax=axs7[1]
ax.plot(v1, ep1, '.', alpha=0.5, label='data')

pfit1 = np.polyfit(v1, ep1, deg=4)
ax.plot(v1, pfit1[0] * v1 ** 4 + pfit1[1] * v1 ** 3 + pfit1[2] * v1 ** 2 + pfit1[3] * v1 + pfit1[4], label='deg=4 fit')

ax.legend()

fig7.savefig(plot_dir + rf'\{targetMZI.up_ps.label}_fit_ep.pdf')


'''Fit V against I'''
fig8, axs8 = plt.subplots(figsize=(10, 5))
ax = axs8

ax.plot(c1, v1, '.', alpha=0.5, label='data')
ax.set_xlabel('Current (mA)')
ax.set_ylabel('Voltage (V)')

pfit1 = np.polyfit(c1 ,v1, deg=2)
ax.plot(c1, pfit1[0] * c1 ** 2 + pfit1[1] * c1 + pfit1[2], label='deg=2 fit')

pfit1 = np.polyfit(c1 ,v1, deg=3)
ax.plot(c1, pfit1[0] * c1 ** 3 + pfit1[1] * c1 ** 2 + pfit1[2] * c1 + pfit1[3], label='deg=3 fit')

pfit1 = np.polyfit(c1, v1, deg=4)
ax.plot(c1, pfit1[0] * c1 ** 4 + pfit1[1] * c1 ** 3 + pfit1[2] * c1 ** 2 + pfit1[3] * c1 + pfit1[4], label='deg=4 fit')

ax.legend()

fig8.savefig(plot_dir + rf'\{targetMZI.up_ps.label}_fit_v_against_i.pdf')


'''Fit op against I'''
def pi_fit_func(I, a, c, b, q1, q2, q3, q4, q5):
    return a + c * np.sin(q1*I + q2*I**2 + q3*I**3 + q4*I**4 + q5*I**5 + b)

p0 = np.array([0.5, 0.5, np.pi/2, 0, 0, 0, 0, 0])
ep_i_fit = np.polyfit(c1, ep1, deg=5)
p0[2:] += ep_i_fit[::-1] * 2*np.pi/ 700

popt1, pcov1 = curve_fit(pi_fit_func, c1, op1/ np.max(op1), p0=p0)

(_, c2, v2, ep2, _, op2) = tuple(mean_results2.T)
p0[2] = np.pi - popt1[2]
popt2, pcov2 = curve_fit(pi_fit_func, c2, op2/np.max(op2), p0=p0)

fig9, axs9 = plt.subplots(2,1, sharex='all')

ax = axs9[0]
ax.plot(c1, op1/np.max(op1), '.', alpha=0.5, label='data')
ax.set_ylabel('Normalised optical power')
ax.set_xlabel('Current (mA)')

cs = np.linspace(0, 65, 800)
ax.plot(cs, pi_fit_func(cs, *popt1), label='fit')
ax.legend()
ax.set_title(f'{targetMZI.up_ps.label}')

ax = axs9[1]
ax.plot(c2, op2/np.max(op2), '.', alpha=0.5, label='data')
ax.set_ylabel('Normalised optical power')
ax.set_xlabel('Current (mA)')

ax.plot(cs, pi_fit_func(cs, *popt2), label='fit')
ax.legend()
ax.set_title(f'{targetMZI.low_ps.label}')

fig9.savefig(plot_dir + rf'\fit_op_vs_i.pdf')

'''Plot set c against c'''
fig10, axs10 = plt.subplots(2,1, figsize=(10,8), layout='constrained', sharex='all')
ax= axs10[0]
ax.plot(set_c1, set_c1, label='y=x')
ax.plot(set_c1, c1, '.', alpha=0.5, label='data')
ax.set_xlabel('Set current (mA)')
ax.set_ylabel('Measured current (mA)')
ax.legend()

ax=axs10[1]
ax.plot(set_c1, c1-set_c1, label='measured - set current (mA)')
ax.legend()
ax.set_ylabel('Current difference (mA)')
ax.set_xlabel('Set current (mA)')

fig10.savefig(plot_dir + rf'\{targetMZI.up_ps.label}_set_c_against_read_c.pdf')