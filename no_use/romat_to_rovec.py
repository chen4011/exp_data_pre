# rotation matrix to rotation vector
import numpy as np
import cv2 as cv
R1_mat = np.array(
    [[-0.99713, 0.00504186, -0.0755413],
    [0.0221672, -0.93461, -0.354982],
    [-0.0723915, -0.355637, 0.931816]])
R2_mat = np.array(
    [[-0.904902, -0.0205024, 0.425125],
    [-0.141438, -0.92759, -0.345793],
    [0.401432, -0.373038, 0.836478]])

R1_vec = cv.Rodrigues(R1_mat)[0]
R2_vec = cv.Rodrigues(R2_mat)[0]
print(R1_vec)
print(R2_vec)