import numpy as np
import matplotlib.pyplot as plt

from prakash.mesh import Mesh
from prakash.plot_utils import CaliUtils

phaseshifter = (0,1)

pMesh = Mesh()

c_results, pp_popt, pi_popt = pMesh.calibrate_from_data(phaseshifter, to_plot=True)

# pMesh.save_calibration()

