import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
from time import sleep

import prakash.config as config
from prakash.driver.optical_switch import LF30CHSM, Hand
from prakash.driver.power_supply import XPOWu
from prakash.driver.power_meter import PM100D
from prakash.mesh import Mesh
from prakash.calibrator import FitCalibrator
from prakash.plot_utils import plot_ps_sweeps, plot_mzi_sweeps, plot_sigma_sweeps
from prakash.utils import DFUtils

print(config.time_stamp)

mesh_name = 'prakash_one'

power_supply = XPOWu(['COM15', 'COM16', 'COM17'])

# Find resource by pyvisa.ResourceManager().list_resources()
opm = PM100D('USB0::0x1313::0x8078::P0022518::INSTR')  # optical power meter

# osw = LF30CHSM('COM4')  # optical switch
osw = Hand()
sleep(1)

pMesh = Mesh(name=mesh_name, power_supply=power_supply)

pCal = FitCalibrator(mesh=pMesh, opm=opm, osw=osw)

k = 1
fitting_results = pCal.calibrate_diag(k=k, save_param=True, averaging=False)
delta_popts = fitting_results['delta']
sigma_popts = fitting_results['sigma']
ex_popts = fitting_results['theta']

# backup calibration data elsewhere
results_dir = config.home_dir + rf'\..\Results\calibration\k={k}_{config.time_stamp}'
pMesh.backup_calibration(file_dir=results_dir)
with open(DFUtils.create_filename(results_dir + r'\popts.pkl'), 'wb') as f:
    pickle.dump(fitting_results, f)

# plot
diags = pMesh.get_diagonal(k=k)
plot_dir = results_dir + rf'\plots'
for mzi_label in diags['mzi']:
    (i,j) = mzi_label
    plot_ps_sweeps((i,j), mesh_name, plot_dir=plot_dir+r'\phaseshifter', pp_popt=delta_popts[(i,j)][0])
    plot_ps_sweeps((i+1,j), mesh_name, plot_dir=plot_dir+r'\phaseshifter', pp_popt=delta_popts[(i,j)][1])

    plot_mzi_sweeps(mzi_label, mesh_name, plot_dir=plot_dir+r'\mzi', popt1=delta_popts[(i,j)][0], popt2=delta_popts[(i,j)][1])

    if j != 0:
        plot_sigma_sweeps(mzi_label, mesh_name, plot_dir=plot_dir+r'\sigma', popt=sigma_popts[mzi_label])
plt.pause(20)

for ps_label in diags['ps']:
    plot_ps_sweeps(ps_label, mesh_name, plot_dir=plot_dir+r'\ex_ps', pp_popt=ex_popts[ps_label])

plt.pause(20)

