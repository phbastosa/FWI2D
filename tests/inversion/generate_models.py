import numpy as np

from scipy.ndimage import gaussian_filter

x_max = 5e3
z_max = 2e3

dh = 10.0

nx = int((x_max / dh) + 1)
nz = int((z_max / dh) + 1)

true_model = np.zeros((nz, nx)) + 2000

true_model[50:] = 2250
true_model[100:] = 2500
true_model[150:] = 3000

init_model = 1.0 / gaussian_filter(1.0 / true_model, 5.0)

true_model.flatten("F").astype(np.float32, order = "F").tofile(f"../inputs/models/inversion_test_true_model_{nz}x{nx}_{dh:.0f}m.bin")
init_model.flatten("F").astype(np.float32, order = "F").tofile(f"../inputs/models/inversion_test_init_model_{nz}x{nx}_{dh:.0f}m.bin")
