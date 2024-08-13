import numpy as np
from scipy.spatial.transform import Rotation as R

def degrees_to_radians(degrees):
    return degrees * np.pi / 180

def quaternion_to_rotation_matrix(q):
    """Convert a quaternion into a rotation matrix."""
    q_w, q_x, q_y, q_z = q
    R = np.array([
        [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
        [2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_x*q_w)],
        [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
    ])
    return R

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return [w, x, y, z]

def get_initial_rotation_matrix(quaternion):
    """Get the initial rotation matrix R0 from the initial quaternion."""
    R_IMU = quaternion_to_rotation_matrix(quaternion)
    R0 = np.linalg.inv(R_IMU)  # Inverse of the rotation matrix
    return R0

bones = [
    # 'Head',
    'Sternum','Pelvis', 'L_UpArm', 'R_UpArm', 'L_LowArm', 'R_LowArm', 'L_UpLeg', 'R_UpLeg', 'L_LowLeg', 'R_LowLeg']

# Angle from east to 284 degrees (counter-clockwise) in radians in ENU
angle_in_NED = 360-0
pitch_deg = 180
# angle_deg = (360 - angle_in_NED + 90) % 360  # Adjusting to ENU
angle_deg = angle_in_NED  # Adjusting to ENU
angle_rad = degrees_to_radians(angle_deg)
pitch_rad = degrees_to_radians(pitch_deg)

# Quaternion representing the rotation from east to 284 degrees in ENU
q_face = [np.cos(angle_rad / 2), 0, 0, np.sin(angle_rad / 2)]
q_pitch = [np.cos(pitch_rad / 2), np.sin(pitch_rad / 2), 0, 0]

# --------- 記得改四元數 ------------
# Example initial quaternion (e.g., when facing 284 degrees)
initial_quaternion = {
    'Sternum':[0.844675, -0.064843, -0.525992, -0.075188],
    'Pelvis':[-0.083004, -0.573975, 0.055253, -0.812779],
    'L_UpArm':[0.830165, 0.180714, -0.052911, 0.524756],
    'R_UpArm':[0.511930, -0.486537, 0.211982, -0.675480],
    'L_LowArm':[-0.749208, 0.027815, 0.269086, -0.604571],
    'R_LowArm':[0.685862, -0.126614, -0.122221, -0.706133],
    'L_UpLeg':[0.502683, 0.405444, -0.651123, 0.398703],
    'R_UpLeg':[0.410653, -0.518245, -0.572393, -0.484926],
    'L_LowLeg':[0.548158, 0.378899, -0.640567, 0.381617],
    'R_LowLeg':[-0.481933, -0.590799, 0.408523, -0.501802]}
# ----------------------------------

R_ib = dict()
with open('s1_walking2_calib_imu_bone_self.txt', 'wb') as f:
    f.write((str(len(initial_quaternion)) + '\n').encode())  # number of bones
    for bone in bones:
        print("Bone:", bone)
        initial_quaternion_bone = initial_quaternion[bone]

        # Apply the correction quaternion to the initial quaternion
        corrected_initial_quaternion = quaternion_multiply(q_face, initial_quaternion_bone)
        corrected_initial_quaternion = quaternion_multiply(q_pitch, corrected_initial_quaternion)

        # Reorder the elements of the quaternion
        q_face_reordered = [q_face[1], q_face[2], q_face[3], q_face[0]]
        initial_quaternion_reordered = [initial_quaternion_bone[1], initial_quaternion_bone[2], initial_quaternion_bone[3], initial_quaternion_bone[0]]

        # Perform the multiplication
        result = (R.from_quat(q_face_reordered) * R.from_quat(initial_quaternion_reordered)).as_quat()

        # # Calculate the corrected initial rotation matrix R0
        # R0_corrected = get_initial_rotation_matrix(corrected_initial_quaternion)

        R_ib[bone] = corrected_initial_quaternion
        # print("corrected_initial_quaternion R0:", corrected_initial_quaternion)
        # print("Corrected initial rotation quaternion R0_corrected:", result)
        f.write((bone + '\t' + str(corrected_initial_quaternion[1]) + '\t' + str(corrected_initial_quaternion[2]) + '\t'
                 + str(corrected_initial_quaternion[3]) + '\t' + str(corrected_initial_quaternion[0]) + '\n').encode())

# print("Corrected initial rotation quaternion R0_corrected:")
# print(R_ib)

# # Example current quaternion at time t from an IMU
# current_quaternion = [0.707, 0.0, 0.707, 0.0]

# # # Apply the initial rotation matrix to transform IMU data to world coordinates
# # def apply_rotation_matrix(R0, quaternion):
# #     """Apply the initial rotation matrix to transform IMU data to world coordinates."""
# #     R_t = quaternion_to_rotation_matrix(quaternion)
# #     R_world = np.dot(R0, R_t)
# #     return R_world

# # Calculate rotation matrix in world coordinates using the corrected R0
# R_world = quaternion_multiply(corrected_initial_quaternion, current_quaternion)
# print("Rotation matrix in world coordinates:")
# print(R_world)