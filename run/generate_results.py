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

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

SPS = np.loadtxt(sps_path, dtype = np.float32, delimiter = ",")
RPS = np.loadtxt(rps_path, dtype = np.float32, delimiter = ",")

xloc = np.linspace(0, (nx-1)*dh, 6)
xlab = np.linspace(0, (nx-1)*dh, 6, dtype = int)

zloc = np.linspace(0, (nz-1)*dh, 5)
zlab = np.linspace(0, (nz-1)*dh, 5, dtype = int)

nr = len(RPS)
ns = len(SPS)

dr = RPS[1,0] - RPS[0,0]

# Modeling results

model_true_file = "../inputs/models/overthrust_true_81x201_25m.bin"

model_true = pyf.read_binary_matrix(nz, nx, model_true_file)

vmin = np.min(model_true)
vmax = np.max(model_true)

sId = int(0.5*ns)

dobs = pyf.read_binary_matrix(nt, nr, f"../inputs/data/seismogram_nt{nt}_nr{nr}_{int(dt*1e6)}us_shot_{sId+1}.bin")

fig, ax = plt.subplots(figsize = (10,3))

ax.imshow(model_true, cmap = "jet", aspect = "auto", extent = [0, (nx-1)*dh, (nz-1)*dh, 0], vmin = vmin, vmax = vmax)

ax.plot(RPS[:,0], RPS[:,1], "o", color = "gray")
ax.plot(SPS[:,0], SPS[:,1], "*", color = "black")

ax.set_xlabel("Distance [m]", fontsize = 15)
ax.set_ylabel("Depth [m]", fontsize = 15)

ax.set_xticks(xloc)
ax.set_xticklabels(xlab)

ax.set_yticks(zloc)
ax.set_yticklabels(zlab)

fig.tight_layout()
plt.savefig("model_true.png", dpi = 200)
plt.show()



scale = np.std(dobs)

offset = np.arange(nr)*dr - SPS[sId,0]

oloc = np.linspace(0, nr-1, 6)
olab = offset[::10]

tloc = np.linspace(0, nt-1, 11)
tlab = np.around(np.linspace(0, (nt-1)*dt, 11), decimals = 1)

fig, ax = plt.subplots(figsize = (6, 8))

ax.imshow(dobs, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

ax.set_xticks(oloc)
ax.set_xticklabels(olab)

ax.set_yticks(tloc)
ax.set_yticklabels(tlab)

ax.set_xlabel("Offset [m]", fontsize = 15)
ax.set_ylabel("Time [s]", fontsize = 15)

fig.tight_layout()
plt.savefig("dobs.png", dpi = 200)
plt.show()



# Migration results

model_init_file = "../inputs/models/overthrust_init_81x201_25m.bin"

model_init = pyf.read_binary_matrix(nz, nx, model_init_file)

fig, ax = plt.subplots(figsize = (10,3))

ax.imshow(model_init, cmap = "jet", aspect = "auto", extent = [0, (nx-1)*dh, (nz-1)*dh, 0], vmin = vmin, vmax = vmax)

ax.plot(RPS[:,0], RPS[:,1], "o", color = "gray")
ax.plot(SPS[:,0], SPS[:,1], "*", color = "black")

ax.set_xlabel("Distance [m]", fontsize = 15)
ax.set_ylabel("Depth [m]", fontsize = 15)

ax.set_xticks(xloc)
ax.set_xticklabels(xlab)

ax.set_yticks(zloc)
ax.set_yticklabels(zlab)

fig.tight_layout()
plt.savefig("model_RTM.png", dpi = 200)
plt.show()



dmig = pyf.read_binary_matrix(nt, nr, f"../inputs/data/input_RTM_seismogram_nt{nt}_nr{nr}_{int(dt*1e6)}us_shot_{sId+1}.bin")

scale = np.std(dmig)

fig, ax = plt.subplots(figsize = (6, 8))

ax.imshow(dmig, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

ax.set_xticks(oloc)
ax.set_xticklabels(olab)

ax.set_yticks(tloc)
ax.set_yticklabels(tlab)

ax.set_xlabel("Offset [m]", fontsize = 15)
ax.set_ylabel("Time [s]", fontsize = 15)

fig.tight_layout()
plt.savefig("dmig_RTM.png", dpi = 200)
plt.show()



image_file = f"../outputs/seismic/RTM_section_{nz}x{nx}.bin"

image = pyf.read_binary_matrix(nz, nx, image_file)

scale = 2.0*np.std(image)

fig, ax = plt.subplots(figsize = (10,3))

ax.imshow(image, cmap = "Greys", aspect = "auto", extent = [0, (nx-1)*dh, (nz-1)*dh, 0], vmin = -scale, vmax = scale)

ax.plot(RPS[:,0], RPS[:,1], "o", color = "gray")
ax.plot(SPS[:,0], SPS[:,1], "*", color = "black")

ax.set_xlabel("Distance [m]", fontsize = 15)
ax.set_ylabel("Depth [m]", fontsize = 15)

ax.set_xticks(xloc)
ax.set_xticklabels(xlab)

ax.set_yticks(zloc)
ax.set_yticklabels(zlab)

fig.tight_layout()
plt.savefig("image_RTM.png", dpi = 200)
plt.show()



# Inversion results

convergence_file = "../outputs/residuo/convergence_5_iterations.txt"

convergence = np.loadtxt(convergence_file, dtype = np.float32)

convergence *= 100.0 / np.max(convergence)

fig, ax = plt.subplots(figsize = (10,4))

ax.plot(convergence, "ok--")

ax.set_title("Convergence map", fontsize = 18)
ax.set_xlabel("Iterations", fontsize = 15)
ax.set_ylabel(r"Residuo L$_2$-norm: $|d^{obs} - d^{cal}|^2_2$")

fig.tight_layout()
plt.savefig("residuo_FWI.png", dpi = 200)
plt.show()



model_pred_file = "../outputs/models/model_FWI_50Hz_81x201.bin"

model_pred = pyf.read_binary_matrix(nz, nx, model_pred_file)

fig, ax = plt.subplots(figsize = (10,3))

ax.imshow(model_pred, cmap = "jet", aspect = "auto", extent = [0, (nx-1)*dh, (nz-1)*dh, 0], vmin = vmin, vmax = vmax)

ax.plot(RPS[:,0], RPS[:,1], "o", color = "gray")
ax.plot(SPS[:,0], SPS[:,1], "*", color = "black")

ax.set_xlabel("Distance [m]", fontsize = 15)
ax.set_ylabel("Depth [m]", fontsize = 15)

ax.set_xticks(xloc)
ax.set_xticklabels(xlab)

ax.set_yticks(zloc)
ax.set_yticklabels(zlab)

fig.tight_layout()
plt.savefig("model_FWI.png", dpi = 200)
plt.show()
