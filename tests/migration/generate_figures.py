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

fig, ax = plt.subplots(figsize = (8, 6))

im = ax.imshow(model, aspect = "auto", cmap = "jet")

cbar = plt.colorbar(im)
cbar.set_label("Velocity P [m/s]", fontsize = 15)

ax.plot(RPS[:,0]/dh, RPS[:,1]/dh, "or", label = "receivers")
ax.plot(SPS[:,0]/dh, SPS[:,1]/dh, "og", label = "sources")
ax.grid(True)
ax.set_xticks(xloc)
ax.set_yticks(zloc)
ax.set_xticklabels(xlab)    
ax.set_yticklabels(zlab)    
ax.set_ylabel("Depth [m]", fontsize = 15)
ax.set_xlabel("Distance [m]", fontsize = 15)

ax.legend(loc = "lower right", fontsize = 15)

fig.tight_layout()
plt.savefig("migration_test_model.png", dpi = 200)

#-----------------------------------------------------------------------------------

image_folder = pyf.catch_parameter(parameters, "mig_output_folder")

image = pyf.read_binary_matrix(nz, nx, image_folder + f"RTM_section_{nz}x{nx}.bin")

image[:int(0.2*nz)] = 0.0

scale = 10*np.std(image)

xloc = np.linspace(0, nx-1, 5)

xlab = np.linspace(0, (nx-1)*dh, 5, dtype = int)

fig, ax = plt.subplots(figsize = (8, 6))

im = ax.imshow(image, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

ax.plot(RPS[:,0]/dh, RPS[:,1]/dh, "or", label = "receivers")
ax.plot(SPS[:,0]/dh, SPS[:,1]/dh, "og", label = "sources")
ax.grid(True)
ax.set_xticks(xloc)
ax.set_yticks(zloc)
ax.set_xticklabels(xlab)    
ax.set_yticklabels(zlab)    
ax.set_ylabel("Depth [m]", fontsize = 15)
ax.set_xlabel("Distance [m]", fontsize = 15)

ax.legend(loc = "lower right", fontsize = 15)

cbar = plt.colorbar(im)
cbar.set_label("Amplitude", fontsize = 15)

fig.tight_layout()
plt.savefig("migration_test_image.png", dpi = 200)
