import numpy as np

from scipy.ndimage import gaussian_filter

x_max = 5e3
z_max = 2e3

dh = 10.0

nx = int((x_max / dh) + 1)
nz = int((z_max / dh) + 1)

model = np.zeros((nz, nx)) + 1500

dv = 50.0
vi = 2000.0

for i in range(nz):
    model[i] = vi + i*dv/dh 

radius = 350

velocity_variation = np.array([500, -500])

circle_centers = np.array([[1250, 1750],
                           [1250, 3250]])

x, z = np.meshgrid(np.arange(nx)*dh, np.arange(nz)*dh)

anomaly = np.zeros_like(model)

for k, dv in enumerate(velocity_variation):
    
    distance = np.sqrt((x - circle_centers[k,1])**2 + (z - circle_centers[k,0])**2)

    anomaly[distance <= radius] += dv

true_model = model + anomaly

true_model.flatten("F").astype(np.float32, order = "F").tofile(f"../inputs/models/inversion_test_true_model_{nz}x{nx}_{dh:.0f}m.bin")

anomaly = gaussian_filter(anomaly, 5.0)

init_model = model + anomaly

init_model.flatten("F").astype(np.float32, order = "F").tofile(f"../inputs/models/inversion_test_init_model_{nz}x{nx}_{dh:.0f}m.bin")
