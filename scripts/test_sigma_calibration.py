import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import json
from time import sleep

import prakash.config as config
from prakash.driver.optical_switch import LF30CHSM
from prakash.driver.power_supply import XPOWu
from prakash.driver.power_meter import PM100D

from prakash.mesh import Mesh
from prakash.calibrator import Calibrator
from prakash.plot_utils import plot_ps_sweeps, plot_mzi_sweeps, plot_sigma_sweeps
from prakash.utils import DFUtils

print(config.time_stamp)

voltage_lim = 15

mesh_name = 'prakash_one'

power_supply = XPOWu(['COM12', 'COM13', 'COM14'])

# Find resource by pyvisa.ResourceManager().list_resources()
opm = PM100D('USB0::0x1313::0x8078::P0022518::INSTR')  # optical power meter

osw = LF30CHSM('COM4')  # optical switch

pMesh = Mesh(name=mesh_name, power_supply=power_supply)
pCal = Calibrator(mesh=pMesh, opm=opm, osw=osw)

sleep(1)
osw.switch(1)
sleep(1)

# Cross prev diagional
mzi5s = pMesh.get_diagonal(k=5)['mzi']
pMesh.set_mzi(mzi5s, delta=0, voltage_lim=voltage_lim)
sleep(1)
pMesh.set_mzi((8,0), delta=np.pi/2, voltage_lim=voltage_lim)
sleep(1)
pMesh.set_mzi((8,2), delta=0, voltage_lim=voltage_lim)
sleep(1)


# ##### Test sigma calibration
mzi_label = (7,5)#
(i,j) = mzi_label

# sweep sigmas
osw.switch(4)
sleep(0.5)

sweep_sigmas = np.linspace(0, 2*np.pi, num=40)
popt = pCal.calibrate_sigma(mzi_label, save_param=False, sweep_sigmas=sweep_sigmas, read_nums=20)

pMesh.set_mzi(mzi_label, delta=np.pi/2, sigma=0)
plot_dir = config.home_dir + rf'\..\Results\test_sigma_cali\2023-11-23'
fig, ax = plot_sigma_sweeps(mzi_label, mesh_name, popt=popt, plot_dir=plot_dir)
plt.pause(10)
np.save(plot_dir+rf'\popt.npy', popt)
# Construct meta MZI with new parameters and check output

pMesh.set_mzi((i - 2, j), delta=np.pi/2, sigma=0)
sleep(0.1)
pMesh.set_mzi((i - 1, j - 1), delta=np.pi/4)
sleep(0.1)
pMesh.set_mzi((i - 1, j + 1), delta=np.pi/4)
sleep(0.1)
pMesh.set_mzi(mzi_label, delta=np.pi/2, sigma=0)
sleep(0.1)
osw.switch(2)
op1 = opm.read() * 1000
osw.switch(3)
sleep(0.5)
op2 = opm.read() * 1000


# ###### Compare delta calibration with and without sigma averaging.
#
# mzi_label = (1,9)
# (i,j) = mzi_label
#
# results_dir = rf'..\Results\compare_av_sigma\{config.time_stamp}\{mzi_label}_without'
# popt1, popt2 = pCal.calibrate_delta((i,j), save_sweep=True, av_mzi_label=None, backup_dir=results_dir)
#
# plot_ps_sweeps((i,j), mesh_name, plot_dir=results_dir, pp_popt=popt1)
# plot_ps_sweeps((i+1,j), mesh_name, plot_dir=results_dir, pp_popt=popt2)
# plot_mzi_sweeps(mzi_label, mesh_name, plot_dir=results_dir, popt1=popt1, popt2=popt2)
#
# with open(DFUtils.create_filename(results_dir + rf'\{(i,j)}popts.pkl'), 'wb') as f:
#     pickle.dump(popt1, f)
# with open(DFUtils.create_filename(results_dir + rf'\{(i+1,j)}popts.pkl'), 'wb') as f:
#     pickle.dump(popt2, f)
#
#
# pCal.ps_sweep_params['read_nums'] = 1
#
# results_dir = rf'..\Results\compare_av_sigma\{config.time_stamp}\{(i,j)}_with'
#
# popt1, popt2 = pCal.calibrate_delta((i,j), save_sweep=True, av_mzi_label=(0,8), backup_dir=results_dir)
#
# plot_ps_sweeps((i,j), mesh_name, plot_dir=results_dir, pp_popt=popt1)
# plot_ps_sweeps((i+1,j), mesh_name, plot_dir=results_dir, pp_popt=popt2)
# plot_mzi_sweeps(mzi_label, mesh_name, plot_dir=results_dir, popt1=popt1, popt2=popt2)
# with open(DFUtils.create_filename(results_dir + rf'\{(i,j)}popts.pkl'), 'wb') as f:
#     pickle.dump(popt1, f)
# with open(DFUtils.create_filename(results_dir + rf'\{(i+1,j)}popts.pkl'), 'wb') as f:
#     pickle.dump(popt2, f)

# ##### Calibrate (2,8)
# mzi_label = (2,8)
# popt1, popt2 = pCal.calibrate_delta(mzi_label)


power_supply.zero_all()