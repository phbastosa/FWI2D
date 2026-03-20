import sys; sys.path.append("../src/")

import numpy as np
import functions as pyf
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

parameters = str(sys.argv[1])

SPS_path = pyf.catch_parameter(parameters,"SPS") 
RPS_path = pyf.catch_parameter(parameters,"RPS") 

model_file = pyf.catch_parameter(parameters,"model_file")

nx = int(pyf.catch_parameter(parameters, "x_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dh = float(pyf.catch_parameter(parameters, "model_spacing"))

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

ns = 6
nr = 51

SPS = np.zeros((ns, 2))
RPS = np.zeros((nr, 2))

SPS[:, 0] = np.linspace(1000, 4000, ns) 
SPS[:, 1] = 100.0 

RPS[:, 0] = np.linspace(0, 5000, nr)
RPS[:, 1] = 0.0 

np.savetxt(SPS_path, SPS, fmt = "%.2f", delimiter = ",")
np.savetxt(RPS_path, RPS, fmt = "%.2f", delimiter = ",")

m_true = np.fromfile(model_file, dtype = np.float32, count = nx*nz).reshape([nz,nx], order = "F")

vmin = np.min(m_true)
vmax = np.max(m_true)

m_init = m_true.copy()

m_init[15:] = 1.0 / gaussian_filter(1.0 / m_true[15:], 3.0)

m_init.flatten("F").astype(np.float32, order = "F").tofile(model_file.replace("true", "init"))

xloc = np.linspace(0, (nx-1)*dh, 6)
xlab = np.linspace(0, (nx-1)*dh, 6, dtype = int)

zloc = np.linspace(0, (nz-1)*dh, 5)
zlab = np.linspace(0, (nz-1)*dh, 5, dtype = int)

fig, ax = plt.subplots(figsize = (10,6), nrows = 2)

ax[0].imshow(m_true, cmap = "jet", aspect = "auto", extent = [0, (nx-1)*dh, (nz-1)*dh, 0], vmin = vmin, vmax = vmax)
ax[1].imshow(m_init, cmap = "jet", aspect = "auto", extent = [0, (nx-1)*dh, (nz-1)*dh, 0], vmin = vmin, vmax = vmax)

for i in range(len(ax)):

    ax[i].plot(RPS[:,0], RPS[:,1], "o", color = "gray")
    ax[i].plot(SPS[:,0], SPS[:,1], "*", color = "black")

    ax[i].set_xlabel("Distance [m]", fontsize = 15)
    ax[i].set_ylabel("Depth [m]", fontsize = 15)

    ax[i].set_xticks(xloc)
    ax[i].set_xticklabels(xlab)

    ax[i].set_yticks(zloc)
    ax[i].set_yticklabels(zlab)

fig.tight_layout()
plt.show()
