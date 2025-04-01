import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

file = str(sys.argv[1])

SPS = np.loadtxt(pyf.catch_parameter(file,"SPS"), delimiter = ",", dtype = np.float32) 
RPS = np.loadtxt(pyf.catch_parameter(file,"RPS"), delimiter = ",", dtype = np.float32) 
XPS = np.loadtxt(pyf.catch_parameter(file,"XPS"), delimiter = ",", dtype = np.int32) 

nx = int(pyf.catch_parameter(file, "x_samples"))
nz = int(pyf.catch_parameter(file, "z_samples")) 

dx = float(pyf.catch_parameter(file, "x_spacing"))
dz = float(pyf.catch_parameter(file, "z_spacing"))

nt = int(pyf.catch_parameter(file, "time_samples"))
dt = float(pyf.catch_parameter(file, "time_spacing"))

model = pyf.read_binary_matrix(nz, nx, pyf.catch_parameter(file, "model_file"))

xloc = np.linspace(0, nx-1, 11)
xlab = np.array(xloc*dx, dtype = int)

zloc = np.linspace(0, nz-1, 6)
zlab = np.array(zloc*dz, dtype = int)

fig, ax = plt.subplots(figsize = (15, 7))

im = ax.imshow(model, aspect = "auto", cmap = "Greys")

cbar = plt.colorbar(im)
cbar.set_label("Velocity P [m/s]")

ax.plot(RPS[:, 0]/dx, RPS[:, 1]/dz, "ob")
ax.plot(SPS[:, 0]/dx, SPS[:, 1]/dz, "or")

ax.set_xticks(xloc)
ax.set_yticks(zloc)
ax.set_xticklabels(xlab)    
ax.set_yticklabels(zlab)    
ax.set_ylabel("Depth [km]", fontsize = 15)
ax.set_xlabel("Distance [km]", fontsize = 15)

fig.tight_layout()
plt.savefig("modeling_test_model.png", dpi = 200)
plt.show()


fmax = float(pyf.catch_parameter(file, "max_frequency"))

nTraces = np.sum(XPS[:,2] - XPS[:,1])

data_folder = pyf.catch_parameter(file, "modeling_output_folder") 

seismic = pyf.read_binary_matrix(nt, nTraces, data_folder + f"seismogram_{int(fmax)}Hz_{nt}x{nTraces}_{int(1e3*dt)}ms.bin")

scale = np.std(seismic)

fig, ax = plt.subplots(figsize = (15, 7))

xloc = np.linspace(0, nTraces-1, 5)
xlab = np.linspace(0, nTraces, 5, dtype = int)

tloc = np.linspace(0, nt-1, 11)
tlab = np.linspace(0, (nt-1)*dt, 11, dtype = float)

im = ax.imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

ax.set_xticks(xloc)
ax.set_yticks(tloc)
ax.set_xticklabels(xlab)    
ax.set_yticklabels(tlab)    
ax.set_ylabel("Time [s]", fontsize = 15)
ax.set_xlabel("Traces", fontsize = 15)

fig.tight_layout()
plt.savefig("modeling_test_data.png", dpi = 200)
plt.show()