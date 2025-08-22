import numpy as np

ns = 4
nr = 9

SPS = np.zeros((ns, 2))
RPS = np.zeros((nr, 2))
XPS = np.zeros((ns, 3))

SPS[:,0] = np.linspace(350, 650, ns)  
SPS[:,1] = np.zeros(ns) + 200 

RPS[:,0] = np.linspace(100, 900, nr) 
RPS[:,1] = np.zeros(nr) + 200 

XPS[:,0] = np.arange(ns)
XPS[:,1] = np.zeros(ns)
XPS[:,2] = np.zeros(ns) + nr

np.savetxt("../inputs/geometry/migration_test_SPS.txt", SPS, fmt = "%.2f", delimiter = ",")
np.savetxt("../inputs/geometry/migration_test_RPS.txt", RPS, fmt = "%.2f", delimiter = ",")
np.savetxt("../inputs/geometry/migration_test_XPS.txt", XPS, fmt = "%.0f", delimiter = ",")