import numpy as np
import matplotlib.pyplot as plt
import pickle

from prakash.mesh import Mesh
from prakash.plot_utils import plot_ps_sweeps, plot_mzi_sweeps, plot_sigma_sweeps
import prakash.config as config

k = 0
if k >= 4:
    final_layer = 9
else:
    final_layer = 0  # reversed

mesh_name = 'prakash_one'
results_dir = config.home_dir + rf'\{mesh_name}'
plot_dir = results_dir + rf'\plots\k={k}'
pMesh = Mesh(mesh_name)

diags = pMesh.get_diagonal(k=k)
for mzi_label in diags['mzi']:
    (i,j) = mzi_label

    try:
        plot_ps_sweeps((i,j), mesh_name, plot_dir=plot_dir+r'\phaseshifter', pp_popt=None)
        plot_ps_sweeps((i+1,j), mesh_name, plot_dir=plot_dir+r'\phaseshifter', pp_popt=None)

        plot_mzi_sweeps(mzi_label, mesh_name, plot_dir=plot_dir+r'\mzi', popt1=None, popt2=None)

        if j != final_layer and k != 4:
            plot_sigma_sweeps(mzi_label, mesh_name, plot_dir=plot_dir+r'\sigma', popt=None)
    except ValueError:
        print(f'No calibration file found for {mzi_label}')
        continue

for ps_label in diags['ps']:
    (i,j) = ps_label

    try:
        if j != final_layer:
            plot_ps_sweeps(ps_label, mesh_name, plot_dir=plot_dir+r'\ex_ps', pp_popt=None)
    except ValueError:
        print(f'No Calibration file found for {ps_label}')
        continue

plt.pause(10)