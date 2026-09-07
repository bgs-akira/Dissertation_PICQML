import numpy as np
import matplotlib.pyplot as plt

from prakash.mesh import Mesh

mzi_label = (2,6)

pMesh= Mesh('prakash_one')


res_dir = r'C:\Users\PhotonsLocalAdmin\Documents\prakash_one_control\Results\test_crosstalk'

data_dirs = {
    (1,6): res_dir + r'\mzi_(2, 6)_2024-03-19(17-21-48.459991)',
    (0, 6): res_dir + r'\mzi_(2, 6)_2024-03-20(17-52-48.874888)'
}

base_file = r'C:\Users\PhotonsLocalAdmin\Documents\prakash_one_control\Results\test_crosstalk\mzi_(2, 6)_2024-03-19(17-21-48.459991)\(2, 6)_c_results_when_(1, 6)=0.00mA.npy'
results_base = np.load(base_file)
mean_base = np.mean(results_base, axis=1)
(set_cbase, cbase, vbase, epbase, _, opbase) = tuple(mean_base.T)

c2s = [0, 30, 45, 60]
colors = ['blue', 'orange', 'green', 'red']
fig, axs = plt.subplots(3,2, figsize=(12, 8), layout='constrained', sharex='col', sharey='row')

for i_col in range(2):
    neighbour = list(data_dirs.keys())[i_col]
    data_dir = data_dirs[neighbour]

    ax1, ax2, ax3 = axs[:, i_col]

    for i_c, c2 in enumerate(c2s):
        if c2 == 0:
            (set_cs, cs, vs, eps, ops) = (set_cbase, cbase, vbase, epbase, opbase)
        else:
            c_results = np.load(data_dir + rf'\{mzi_label}_c_results_when_{neighbour}={c2:.2f}mA.npy')
            mean_results = np.mean(c_results, axis=1)
            (set_cs, cs, vs, eps, _, ops) = tuple(mean_results.T)

        plot_params = {'label': f'c2={c2}mA', 'alpha': 0.5, 'color': colors[i_c]}
        ax1.plot(set_cs, ops, **plot_params)
        ax2.plot(set_cs, ops - opbase, **plot_params)
        ax3.plot(set_cs, cs - cbase, **plot_params)



    ax1.legend()
    ax1.set_title(f'With heat on {neighbour} (ch{pMesh.get_channel(neighbour)})')
    ax2.set_title('Change in optical power')
    ax3.set_title('Change in measured current')
    ax3.set_xlabel('Set current (mA)')


    if i_col == 0:
        ax1.set_ylabel('Optical power (mW)')
        ax2.set_ylabel('Optical power (mW)')
        ax3.set_ylabel('Current (mA)')

fig.suptitle(f'Sweeping Phaseshifter {mzi_label} (ch{pMesh.get_channel(mzi_label)})')

fig.savefig(data_dir + r'\Comparison_crosstalk.pdf')