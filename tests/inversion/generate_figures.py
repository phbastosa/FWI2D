import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

parameters = str(sys.argv[1])

SPS = np.loadtxt(pyf.catch_parameter(parameters,"SPS"), delimiter = ",", dtype = np.float32) 
RPS = np.loadtxt(pyf.catch_parameter(parameters,"RPS"), delimiter = ",", dtype = np.float32) 
XPS = np.loadtxt(pyf.catch_parameter(parameters,"XPS"), delimiter = ",", dtype = np.int32) 

nx = int(pyf.catch_parameter(parameters, "x_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dx = float(pyf.catch_parameter(parameters, "x_spacing"))
dz = float(pyf.catch_parameter(parameters, "z_spacing"))

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

true_model = pyf.read_binary_matrix(nz, nx, "../inputs/models/inversion_test_true_model_201x501_10m.bin")
init_model = pyf.read_binary_matrix(nz, nx, "../inputs/models/inversion_test_init_model_201x501_10m.bin")

xloc = np.linspace(0, nx-1, 11)
xlab = np.array(xloc*dx, dtype = int)

zloc = np.linspace(0, nz-1, 6)
zlab = np.array(zloc*dz, dtype = int)

fig, ax = plt.subplots(nrows = 2, figsize = (15, 8))

im = ax[0].imshow(true_model, aspect = "auto", cmap = "jet")

cbar = plt.colorbar(im)
cbar.set_label("Velocity P [m/s]", fontsize = 15)

ax[0].plot(RPS[:, 0]/dx, RPS[:, 1]/dz, "ob")
ax[0].plot(SPS[:, 0]/dx, SPS[:, 1]/dz, "or")

ax[0].set_xticks(xloc)
ax[0].set_yticks(zloc)
ax[0].set_xticklabels(xlab)    
ax[0].set_yticklabels(zlab)    
ax[0].set_ylabel("Depth [km]", fontsize = 15)
ax[0].set_xlabel("Distance [km]", fontsize = 15)

im = ax[1].imshow(init_model, aspect = "auto", cmap = "jet")

cbar = plt.colorbar(im)
cbar.set_label("Velocity P [m/s]", fontsize = 15)

ax[1].plot(RPS[:, 0]/dx, RPS[:, 1]/dz, "ob")
ax[1].plot(SPS[:, 0]/dx, SPS[:, 1]/dz, "or")

ax[1].set_xticks(xloc)
ax[1].set_yticks(zloc)
ax[1].set_xticklabels(xlab)    
ax[1].set_yticklabels(zlab)    
ax[1].set_ylabel("Depth [km]", fontsize = 15)
ax[1].set_xlabel("Distance [km]", fontsize = 15)

fig.tight_layout()
plt.savefig("inversion_test_models.png", dpi = 200)

