import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress

from prakash.mesh import Mesh
from prakash.utils import DFUtils

'''Parameters'''
mesh_name = 'prakash_one'
mzi_label = (1, 7)
i, j = mzi_label

pMesh = Mesh(name=mesh_name, power_supply=None)
targetMZI = pMesh.mzi(mzi_label)

'''Load data'''
data_dir = rf'/Results/test_delta_sweep/mzi_(1, 7)_2024-03-11(15-11-45.490395)'

results1 = np.load(data_dir + rf'\{i, j}_c_results.npy')
mean1 = np.mean(results1, axis=1)
(set_c1, c1, v1, ep1, _, op1) = tuple(mean1.T)

results2 = np.load(data_dir + rf'\{i + 1, j}_c_results.npy')
mean2 = np.mean(results2, axis=1)
(set_c2, c2, v2, ep2, _, op2) = tuple(mean2.T)

# from optimising
pmax = 4.2
pmin = 6.01e-5

'''Function to calculate phase'''
def calc_phase(p, pmax, pmin):
    p = np.atleast_1d(p)
    ratios = (2 * p - pmin - pmax) / (pmax - pmin)

    ratios[np.argwhere(ratios > 1.)] = 1.
    ratios[np.argwhere(ratios < -1.)] = -1.

    return np.arcsin(ratios)  # only between -pi/2 to pi/2. Need to adjust afterwards.

def correct_phase(phi, imins, imaxs):
    imins = np.atleast_1d(imins)
    imaxs = np.atleast_1d(imaxs)

    new_phi = np.zeros_like(phi)
    if len(imaxs) == 1:
        imaxs = np.concatenate([np.array([0]), imaxs])

    new_phi[:imaxs[0]] = phi[:imaxs[0]]
    new_phi[imaxs[0]: imins[0]] = - phi[imaxs[0]: imins[0]]
    new_phi[imins[0]: imaxs[1]] = phi[imins[0]: imaxs[1]] + 2 * np.pi
    new_phi[imaxs[1]:] = 2 * np.pi - phi[imaxs[1]:]

    return new_phi

# test_b1 = calc_phase(op1[0], pmax, pmin)[0]
# test_b2 = calc_phase(op1[0], np.max(op1), np.min(op1))[0]
# test_b3 = calc_phase(op2[0], pmax, pmin)[0]
# test_b4 = calc_phase(op2[0], np.max(op2), np.min(op2))[0]

'''Plot sweeps'''
fig, axs = plt.subplots(3, 2, figsize=(10, 12), layout='constrained')
ax = axs[0,0]
ax.plot(ep1, op1, '.', markersize=4, label='data')
ax.set_title(f'Phaseshifter {(i, j)}')
ax.set_xlabel(r'$x$/mW')
ax.set_ylabel(r'$P$/mW')

i_min1 = np.argmin(op1)
ax.plot(ep1[i_min1], op1[i_min1], 'r.', markersize=8, label='min data')
i_maxs1, _ = find_peaks(op1, distance=i_min1)
ax.plot(ep1[i_maxs1], op1[i_maxs1], '.', color='yellow', markersize=8, label='max data')

ax = axs[0, 1]
ax.plot(ep2, op2, '.', markersize=4, label='data')
ax.set_title(f'Phaseshifter {(i + 1, j)}')
ax.set_xlabel(r'$x$/mW')
ax.set_ylabel(r'$P$/mW')

i_min2 = np.argmin(op2)
ax.plot(ep2[i_min2], op2[i_min2], 'r.', markersize=8, label='min data')
i_maxs2, _ = find_peaks(op2, distance=i_min2)
ax.plot(ep2[i_maxs2], op2[i_maxs2], '.', color='yellow', markersize=8, label='max data')

'''Calculate phase directly'''
assert len(i_maxs1) == 2
phi1_1 = calc_phase(op1[:i_min1+1], op1[i_maxs1[0]], np.min([op1[i_min1], pmin])) - np.pi /2
phi1_2 = calc_phase(op1[i_min1+1:], op1[i_maxs1[1]], np.min([op1[i_min1], pmin])) - np.pi /2

phi1 = np.concatenate([phi1_1, phi1_2])

phi2 = calc_phase(op2, np.max([np.max(op2), pmax]), np.min([np.min(op2), pmin])) - np.pi / 2

# TODO: find a better way to write this, mate.


ax = axs[1, 0]
ax.plot(ep1, phi1, '.', markersize=4, label='uncorrected', alpha=0.5)
ax.set_xlabel(r'$x$/mW')
ax.set_ylabel(r'$\phi$')
ax.set_title(f'Phaseshifter {(i, j)}')
ax.set_yticks(-np.linspace(0, np.pi, 5))
ax.plot(ep1, np.zeros_like(ep1), 'k--', alpha=0.5)
ax.plot(ep1, -np.pi * np.ones_like(ep1), 'k--', alpha=0.5)

ax = axs[1,1]
ax.plot(ep2, phi2, '.', markersize=4, label='uncorrected', alpha=0.5)
ax.set_xlabel(r'$x$/mW')
ax.set_ylabel(r'$\phi$')
ax.set_title(f'Phaseshifter {(i + 1, j)}')
ax.set_yticks(-np.linspace(0, np.pi, 5))
ax.plot(ep2, np.zeros_like(ep2), 'k--', alpha=0.5)
ax.plot(ep2, -np.pi * np.ones_like(ep2), 'k--', alpha=0.5)

''' Correct for phases'''
new_phi1 = correct_phase(phi1, i_min1, i_maxs1)
new_phi2 = correct_phase(phi2, i_min2, i_maxs2)


'''Plot corrected phases'''
ax = axs[1,0]
ax.plot(ep1, new_phi1, '.', markersize=4, label='corrected', alpha=0.5)
ax.set_yticks(np.linspace(-np.pi, 2*np.pi, 7))
ax.legend()
ax.plot(ep1, 2*np.pi * np.ones_like(ep1), 'k--', alpha=0.5)

ax = axs[1,1]
ax.plot(ep2, new_phi2, '.', markersize=4, label='corrected', alpha=0.5)
ax.set_yticks(np.linspace(-np.pi, 2*np.pi, 7))
ax.legend()
ax.plot(ep1, 2*np.pi * np.ones_like(ep1), 'k--', alpha=0.5)


'''Linear regression'''
regress1 = linregress(ep1, new_phi1)
ax = axs[1,0]
ax.plot(ep1, regress1.slope * ep1 + regress1.intercept, label='linear regression')
ax.legend()

regress2 = linregress(ep2, new_phi2)
ax = axs[1,1]
ax.plot(ep2, regress2.slope * ep2 + regress2.intercept, label='linear regression')
ax.legend()

b1 = calc_phase(op1[0], pmax, pmin)[0] - np.pi/2
b2 = np.pi/2 - calc_phase(op2[0], pmax, pmin)[0]


'''Phase to current'''
ax = axs[2,0]
ax.plot(set_c1, new_phi1, '.', markersize=4, alpha=0.5, label='data')
ax.set_title(f'Phaseshifter {(i,j)}')
ax.set_ylabel('Phase')
ax.set_xlabel('Set current (mA)')

ax = axs[2,1]
ax.plot(set_c2, new_phi2, '.', markersize=4, alpha=0.5, label='data')
ax.set_title(f'Phaseshifter {(i+1,j)}')

ax.set_ylabel('Phase')
ax.set_xlabel('Set current (mA)')

fig.savefig(DFUtils.create_filename(data_dir + rf'\phase_calibration\plots.pdf'))

'''Save phase data'''
phase_data1 = np.vstack((set_c1, new_phi1))
np.save(data_dir + rf'\phase_calibration\phaseshifter_{(i,j)}_phase_data.npy', phase_data1)

phase_data2 = np.vstack((set_c2, new_phi2))
np.save(data_dir + rf'\phase_calibration\phaseshifter_{(i+1,j)}_phase_data.npy', phase_data2)

'''Interpolation'''
dense_phi1 =np.linspace(np.min(new_phi1), np.max(new_phi1), 10000)
dense_phi2 =np.linspace(np.min(new_phi2), np.max(new_phi2), 10000)
c1_interp = np.interp(dense_phi1, new_phi1, set_c1)
c2_interp = np.interp(dense_phi2, new_phi2, set_c2)

fig2, axs2 = plt.subplots(2,1, figsize=(10, 8), layout='constrained')

ax = axs2[0]
ax.plot(new_phi1, set_c1, '.', markersize=4, alpha=0.5, label='data')
ax.plot(dense_phi1, c1_interp, linewidth=1, label='interp')
ax.set_title(f'Phaseshifter {(i,j)}')
ax.set_ylabel('Set current (mA)')
ax.set_xlabel('Phase')
ax.legend()

ax = axs2[1]
ax.plot(new_phi2, set_c2, '.', markersize=4, alpha=0.5, label='data')
ax.plot(dense_phi2, c2_interp, linewidth=1, label='interp')
ax.set_title(f'Phaseshifter {(i+1,j)}')
ax.set_ylabel('Set current (mA)')
ax.set_xlabel('Phase')
ax.legend()

fig2.savefig(data_dir + rf'\phase_calibration\phase_interpolation.pdf')