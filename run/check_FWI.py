import sys; sys.path.append("../src/")

import numpy as np
import functions as pyf
import matplotlib.pyplot as plt

parameters = str(sys.argv[1])

SPS_path = pyf.catch_parameter(parameters,"SPS") 
RPS_path = pyf.catch_parameter(parameters,"RPS") 

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

SPS = np.loadtxt(SPS_path, dtype = np.float32, delimiter = ",")
RPS = np.loadtxt(RPS_path, dtype = np.float32, delimiter = ",")

nr = len(RPS)

dobs = pyf.read_binary_matrix(nt, nr, f"../inputs/data/obs_seismogram_nt2001_nr51_1000us_shot_1.bin")
dcal = pyf.read_binary_matrix(nt, nr, f"../outputs/data/cal_seismogram_nt2001_nr51_1000us_shot_1.bin")


scale = np.std(dobs)

fig, ax = plt.subplots(figsize = (15,8), ncols = 3)

ax[0].imshow(dobs, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

ax[1].imshow(dcal, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

ax[2].imshow(dobs - dcal, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

fig.tight_layout()
plt.show()



