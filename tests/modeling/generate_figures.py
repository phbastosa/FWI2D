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

dh = float(pyf.catch_parameter(parameters, "model_spacing"))

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

model = pyf.read_binary_matrix(nz, nx, pyf.catch_parameter(parameters, "model_file"))

xloc = np.linspace(0, nx-1, 11)
xlab = np.array(xloc*dh, dtype = int)

zloc = np.linspace(0, nz-1, 6)
zlab = np.array(zloc*dh, dtype = int)

fig, ax = plt.subplots(figsize = (15, 7))

im = ax.imshow(model, aspect = "auto", cmap = "jet")

cbar = plt.colorbar(im)
cbar.set_label("Velocity P [m/s]")

ax.plot(RPS[:, 0]/dh, RPS[:, 1]/dh, "ob")
ax.plot(SPS[:, 0]/dh, SPS[:, 1]/dh, "or")

ax.set_xticks(xloc)
ax.set_yticks(zloc)
ax.set_xticklabels(xlab)    
ax.set_yticklabels(zlab)    
ax.set_ylabel("Depth [km]", fontsize = 15)
ax.set_xlabel("Distance [km]", fontsize = 15)

fig.tight_layout()
plt.savefig("modeling_test_model.png", dpi = 200)



spread = XPS[0,2] - XPS[0,1]

fmax = float(pyf.catch_parameter(parameters, "max_frequency"))

data_folder = pyf.catch_parameter(parameters, "mod_output_folder") 

fig, ax = plt.subplots(ncols = len(SPS), figsize = (15, 7))

xloc = np.linspace(0, spread-1, 5)
xlab = np.linspace(0, spread, 5, dtype = int)

tloc = np.linspace(0, nt-1, 11)
tlab = np.linspace(0, (nt-1)*dt, 11, dtype = float)

for sId in range(len(SPS)):

    template =  f"seismogram_nt{nt}_nr{spread}_{int(1e6*dt)}us_shot_{sId+1}.bin"
    seismic = pyf.read_binary_matrix(nt, spread, data_folder + template)

    scale = np.std(seismic)

    im = ax[sId].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    ax[sId].set_xticks(xloc)
    ax[sId].set_yticks(tloc)
    ax[sId].set_xticklabels(xlab)    
    ax[sId].set_yticklabels(tlab)    
    ax[sId].set_ylabel("Time [s]", fontsize = 15)
    ax[sId].set_xlabel("Traces", fontsize = 15)

fig.tight_layout()
plt.savefig("modeling_test_data.png", dpi = 200)
