import sys; sys.path.append("../src/")

import numpy as np
import functions as pyf

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

ns = 11
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

zmask = float(pyf.catch_parameter(parameters, "depth_mask"))

zId = int(zmask/dh)

m_init = 1.0 / gaussian_filter(1.0 / m_true, 3.0)

m_init[:zId] = m_true[:zId]

m_init.flatten("F").astype(np.float32, order = "F").tofile(model_file.replace("true", "init"))
