import sys; sys.path.append("../src/")

import numpy as np
import functions as pyf
import matplotlib.pyplot as plt

parameters = str(sys.argv[1])

sps_path = pyf.catch_parameter(parameters, "SPS") 
rps_path = pyf.catch_parameter(parameters, "RPS") 

nx = int(pyf.catch_parameter(parameters, "x_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dh = float(pyf.catch_parameter(parameters, "model_spacing"))

model_true_file = "../inputs/models/overthrust_true_81x201_25m.bin"
model_init_file = "../inputs/models/overthrust_init_81x201_25m.bin"
model_pred_file = "../inputs/"

convergence_file = "../outputs/"

image_before_FWI_file = f"../outputs/seismic/RTM_section_{nz}x{nx}.bin"
image_after_FWI_file = f"../outputs/seismic/RTM_section_{nz}x{nx}.bin"




# image = pyf.read_binary_matrix(nz, nx, image_file)
# model = pyf.read_binary_matrix(nz, nx, model_file)

# SPS = np.loadtxt(sps_path, dtype = np.float32, delimiter = ",")
# RPS = np.loadtxt(rps_path, dtype = np.float32, delimiter = ",")

# xloc = np.linspace(0, (nx-1)*dh, 6)
# xlab = np.linspace(0, (nx-1)*dh, 6, dtype = int)

# zloc = np.linspace(0, (nz-1)*dh, 5)
# zlab = np.linspace(0, (nz-1)*dh, 5, dtype = int)

# vmin = np.min(model)
# vmax = np.max(model)

# image *= 1.0 / np.max(np.abs(image))

# scale = 2.0*np.std(image)

# fig, ax = plt.subplots(figsize = (10,6), nrows = 2)

# ax[0].imshow(model, cmap = "jet", aspect = "auto", extent = [0, (nx-1)*dh, (nz-1)*dh, 0], vmin = vmin, vmax = vmax)
# ax[1].imshow(image, cmap = "Greys", aspect = "auto", extent = [0, (nx-1)*dh, (nz-1)*dh, 0], vmin =-scale, vmax = scale)

# for i in range(len(ax)):

#     ax[i].plot(RPS[:,0], RPS[:,1], "o", color = "gray")
#     ax[i].plot(SPS[:,0], SPS[:,1], "*", color = "black")
    
#     ax[i].set_xlabel("Distance [m]", fontsize = 15)
#     ax[i].set_ylabel("Depth [m]", fontsize = 15)

#     ax[i].set_xticks(xloc)
#     ax[i].set_xticklabels(xlab)

#     ax[i].set_yticks(zloc)
#     ax[i].set_yticklabels(zlab)

# fig.tight_layout()
# plt.show()
