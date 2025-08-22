import numpy as np
import matplotlib.pyplot as plt

x_max = 5e3
z_max = 5e3

dh = 10.0

nx = int((x_max / dh) + 1)
nz = int((z_max / dh) + 1)

Vp = np.zeros((nz,nx)) + 1500

hx = int(0.50*nx)
hz = int(4000/dh)

Vp[hz-10:hz+10,:] += 500

Vp.flatten("F").astype(np.float32, order = "F").tofile(f"../inputs/models/migration_test_vp.bin")
