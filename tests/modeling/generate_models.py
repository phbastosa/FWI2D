import numpy as np

x_max = 1e4
z_max = 5e3

dh = 10.0

nx = int((x_max / dh) + 1)
nz = int((z_max / dh) + 1)

Vp = np.zeros((nz, nx)) + 1500

v = np.array([1500, 1700, 1900, 2300, 3000, 3500])
z = np.array([200, 500, 1000, 1500, 1500])

for i in range(len(z)):
    Vp[int(np.sum(z[:i+1]/dh)):] = v[i+1]

Vp[300:350,600:700] += 500

Vp.flatten("F").astype(np.float32, order = "F").tofile(f"../inputs/models/modeling_test_vp_{nz}x{nx}_{dh:.0f}m.bin")
