import sys; sys.path.append("../src/")

import numpy as np
import functions as pyf
import matplotlib.pyplot as plt

parameters = str(sys.argv[1])

sps_path = pyf.catch_parameter(parameters,"SPS") 
rps_path = pyf.catch_parameter(parameters,"RPS") 

nx = int(pyf.catch_parameter(parameters, "x_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dh = float(pyf.catch_parameter(parameters, "model_spacing"))

image_file = "../outputs/seismic/RTM_section_81x201.bin"
model_file = pyf.catch_parameter(parameters,"model_file")

image = pyf.read_binary_matrix(nz,nx,image_file)
model = pyf.read_binary_matrix(nz,nx,model_file)

SPS = np.loadtxt(sps_path, dtype = np.float32, delimiter = ",")
RPS = np.loadtxt(rps_path, dtype = np.float32, delimiter = ",")

scale = 5.0*np.std(image)

fig, ax = plt.subplots(figsize = (15,5), ncols = 2)

ax[0].imshow(model, aspect = "auto", cmap = "jet", extent = [0, (nx-1)*dh, (nz-1)*dh, 0])
ax[0].set_xlabel("Distance [m]", fontsize = 15)
ax[0].set_ylabel("Depth [m]", fontsize = 15)
ax[0].plot(RPS[:,0], RPS[:,1], "o", color = "gray")
ax[0].plot(SPS[:,0], SPS[:,1], "*", color = "green")

ax[1].imshow(image, aspect = "auto", cmap = "Greys", vmin =-scale, vmax = scale, extent = [0, (nx-1)*dh, (nz-1)*dh, 0])
ax[1].set_xlabel("Distance [m]", fontsize = 15)
ax[1].set_ylabel("Depth [m]", fontsize = 15)
ax[1].plot(RPS[:,0], RPS[:,1], "o", color = "gray")
ax[1].plot(SPS[:,0], SPS[:,1], "*", color = "green")

fig.tight_layout()
plt.show()
