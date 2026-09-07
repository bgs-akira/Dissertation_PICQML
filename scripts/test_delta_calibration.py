import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import json
from time import sleep

import prakash.config as config
from prakash.driver.optical_switch import LF30CHSM, Hand
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

# osw = LF30CHSM('COM4')  # optical switch
osw = Hand()

pMesh = Mesh(name=mesh_name, power_supply=power_supply)
pCal = Calibrator(mesh=pMesh, opm=opm, osw=osw)

sleep(1)
osw.switch(1)
sleep(1)

# MZI
mzi_label = (0, 0)
i, j = mzi_label
k = (i + j) // 2

pMesh.set_mzi([(0, 8), (0, 6), (0,4)], delta=np.pi / 2)

prev_comp = pMesh.get_diagonal(k + 1)
# pMesh.set_mzi(prev_comp['mzi'], delta=0)
sleep(0.5)
pMesh.set_mzi((0, 2), delta=np.pi / 2)

# Calibrate
popt1, popt2 = pCal.calibrate_delta(mzi_label, save_param=True)

results_dir = config.home_dir + rf'\..\Results\calibration\k={k}_{config.time_stamp}'

np.save(DFUtils.create_filename(results_dir + rf'\popt_{(i,j)}.npy'), popt1)
np.save(results_dir + rf'\popt_{(i+1,j)}.npy', popt2)

plot_ps_sweeps((i, j), mesh_name, pp_popt=popt1, plot_dir=results_dir+rf'\plot')
plot_ps_sweeps((i+1, j), mesh_name, pp_popt=popt2, plot_dir=results_dir+rf'\plot')
plot_mzi_sweeps(mzi_label, mesh_name, popt1=popt1, popt2=popt2, plot_dir=results_dir+rf'\plot')

# np.save(DFUtils.create_filename(results_dir + rf'\popt_{(0,2)}.npy'), popt1)
# np.save(results_dir + rf'\popt_{(1,2)}.npy', popt2)
# np.save(results_dir + rf'\popt_{(2,0)}.npy', popt3)
# np.save(results_dir + rf'\popt_{(3,0)}.npy', popt4)
#
# plot_ps_sweeps((0,2), mesh_name, pp_popt=popt1, plot_dir=results_dir + rf'\plot')
# plot_ps_sweeps((1,2), mesh_name, pp_popt=popt2, plot_dir=results_dir + rf'\plot')
# plot_mzi_sweeps(mzi_label, mesh_name, popt1=popt1, popt2=popt2, plot_dir=results_dir + rf'\plot')
#
# plot_ps_sweeps((2,0), mesh_name, pp_popt=popt3, plot_dir=results_dir + rf'\plot')
# plot_ps_sweeps((3,0), mesh_name, pp_popt=popt4, plot_dir=results_dir + rf'\plot')
# plot_mzi_sweeps((2,0), mesh_name, popt1=popt3, popt2=popt4, plot_dir=results_dir + rf'\plot')

power_supply.zero_all()
