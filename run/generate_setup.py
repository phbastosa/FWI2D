import sys; sys.path.append("../src/")

import numpy as np
import functions as pyf
import matplotlib.pyplot as plt

parameters = str(sys.argv[1])

SPS_path = pyf.catch_parameter(parameters,"SPS") 
RPS_path = pyf.catch_parameter(parameters,"RPS") 
XPS_path = pyf.catch_parameter(parameters,"XPS")

model_file = pyf.catch_parameter(parameters,"model_file")

nx = int(pyf.catch_parameter(parameters, "x_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dh = float(pyf.catch_parameter(parameters, "model_spacing"))

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

vp = np.array([1500, 1800, 2000])
z = np.array([750, 1000])

ns = 6
nr = 201

SPS = np.zeros((ns, 2))
RPS = np.zeros((nr, 2))
XPS = np.zeros((ns, 3))

SPS[:, 0] = np.linspace(1000, 4000, ns) 
SPS[:, 1] = 100.0 

RPS[:, 0] = np.linspace(0, 5000, nr)
RPS[:, 1] = 0.0 

XPS[:, 0] = np.arange(ns)
XPS[:, 1] = np.zeros(ns) 
XPS[:, 2] = np.zeros(ns) + nr 

np.savetxt(SPS_path, SPS, fmt = "%.2f", delimiter = ",")
np.savetxt(RPS_path, RPS, fmt = "%.2f", delimiter = ",")
np.savetxt(XPS_path, XPS, fmt = "%.0f", delimiter = ",")

Vp = np.zeros((nz,nx))

for i in range(len(vp)):
    layer = int(np.sum(z[:i])/dh)
    Vp[layer:] = vp[i]

Vp.flatten("F").astype(np.float32, order = "F").tofile(model_file)

fig, ax = plt.subplots(figsize = (10,4))

ax.imshow(Vp, cmap = "jet", aspect = "auto", extent = [0, (nx-1)*dh, (nz-1)*dh, 0])

ax.plot(RPS[:,0], RPS[:,1], "o", color = "gray")
ax.plot(SPS[:,0], SPS[:,1], "*", color = "green")

fig.tight_layout()
plt.show()
