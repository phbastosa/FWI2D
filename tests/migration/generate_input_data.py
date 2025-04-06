import sys; sys.path.append("../src/")

import numpy as np
import functions as pyf

parameters = str(sys.argv[1])

XPS = np.loadtxt(pyf.catch_parameter(parameters, "XPS"), delimiter = ",", dtype = np.int32) 

nTraces = (XPS[0,2] - XPS[0,1])*len(XPS)

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

fmax = float(pyf.catch_parameter(parameters, "max_frequency"))

data_folder = pyf.catch_parameter(parameters, "migration_input_folder")

template =  f"seismogram_nt{nt}_nTraces{nTraces}_{int(fmax)}Hz_{int(1e3*dt)}ms.bin"

data_file = data_folder + template

data = pyf.read_binary_matrix(nt, nTraces, data_file)

data[:4500] = 0.0

data.flatten("F").astype(np.float32, order = "F").tofile(data_file)
