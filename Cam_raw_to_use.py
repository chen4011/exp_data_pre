## 將相機校正的原始資料轉換成 cal 檔案(排列成 data fusion 可用的樣式)

import toml
import numpy as np
import cv2 as cv
import os
from datetime import datetime

# 獲取當前日期和時間
current_datetime = datetime.now()
# date = current_datetime.strftime("%Y%m%d")
# date = f'20240616'

# 獲得目前的工作目錄
base_path = os.getcwd()
# imu_raw_path = os.path.join(base_path, 'imu_raw_data')
cam_raw_path = os.path.join(base_path, 'calib_raw_data')
result_path = os.path.join(base_path, 'calib_result_data')

# # 讀取 IMU 文件
# imu_raw_file_name = f'MT_012102F3_029-000_00B48A67.txt'
# with open(os.path.join(imu_raw_path,imu_raw_file_name), 'r') as file:
#     lines = file.readlines()
#     device_id = None
#     for line in lines:
#         if line.strip().startswith("//  DeviceId:"):
#             device_id = line.split(":")[1].strip()
#         elif line.strip().startswith("PacketCounter"):
#             board_orn = lines[lines.index(line) + 1:]
#     print(board_orn)

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

# Angle from east to 284 degrees (counter-clockwise) in radians in ENU
roll_deg = 90
pitch_deg = 180
# angle_deg = (360 - angle_in_NED + 90) % 360  # Adjusting to ENU
roll_rad = degrees_to_radians(roll_deg)
pitch_rad = degrees_to_radians(pitch_deg)

# Quaternion representing the rotation from east to 284 degrees in ENU
q_roll = [np.cos(roll_rad / 2), 0, np.sin(roll_rad / 2), 0]
q_pitch = [np.cos(pitch_rad / 2), np.sin(pitch_rad / 2), 0, 0]

# --------- 記得改四元數 ------------
# board to ENU[w, x, y, z]
board_orn = (0.449063, -0.439756, -0.542420, 0.557438)
# ----------------------------------

# Apply the correction quaternion to the initial quaternion
corrected_initial_quaternion = quaternion_multiply(q_roll, board_orn)
corrected_initial_quaternion = quaternion_multiply(q_pitch, corrected_initial_quaternion)
R_cb2enu = np.eye(4)
R_cb2enu[:3, :3] = quaternion_to_rotation_matrix(corrected_initial_quaternion)
R_enu2cb = np.linalg.inv(R_cb2enu)
# print(R_enu2cb)

# 讀取 TOML 文件
cam_raw_file_name = f'Calib_intcheckerboard_extcheckerboard.toml'
with open(os.path.join(cam_raw_path, cam_raw_file_name), 'r') as f:
    cam_data = toml.load(f)
# print(cam_data)

# 提取並排列需要的資訊及格式
fx, fy, cx, cy = [], [], [], []
distor_param, r1, r2, r3, translation, t = [], [], [], [], [], []
cam_nb = len(cam_data) - 1
for i in range(cam_nb):
    cam_key = f'cam_0{i+1}'

    # camera intrinsic parameters
    fx.append(cam_data[cam_key]['matrix'][0][0])
    fy.append(cam_data[cam_key]['matrix'][1][1])
    cx.append(cam_data[cam_key]['matrix'][0][2])
    cy.append(cam_data[cam_key]['matrix'][1][2])

    # camera distortion parameters
    distor_param.append(cam_data[cam_key]['distortions'])

    # camera extrinsic parameters
    R_vec = cam_data[cam_key]['rotation']
    R_mat = cv.Rodrigues(np.array(R_vec))[0]    # 旋轉向量轉矩陣
    # print(R_mat)
    translation.append(cam_data[cam_key]['translation'])

    # 將 rotation 及 translation 組合成 4*4 的矩陣
    T_cb2cam = np.eye((4))
    T_cb2cam[:3, :3] = R_mat
    T_cb2cam[:3, 3] = translation[i]
    # print(T_cb2cam)

    # 將 camera extrinsic parameters 原本為 board to camera 轉換成 ENU to camera
    T_enu2cam = T_cb2cam @ R_enu2cb
    # print(T_enu2cam)

    r1.append(T_enu2cam[0, :3])
    r2.append(T_enu2cam[1, :3])
    r3.append(T_enu2cam[2, :3])
    t.append(T_enu2cam[:3, 3])
    

# 將資訊寫入 cal 檔案
calibration_file_name = f'calibration_self.cal'
with open(os.path.join(result_path, calibration_file_name),'w') as f:
    f.write(f'{cam_nb} 1\n')  # 固定的行数和列数
    for i in range(cam_nb):
        f.write('0 1079 0 1919\n')
        f.write(f'{fx[i]} {fy[i]} {cx[i]} {cy[i]}\n')
        f.write(f'{distor_param[i][0]} {distor_param[i][1]} {distor_param[i][2]} {distor_param[i][3]}\n')
        f.write(f'{r1[i][0]} {r1[i][1]} {r1[i][2]}\n')
        f.write(f'{r2[i][0]} {r2[i][1]} {r2[i][2]}\n')
        f.write(f'{r3[i][0]} {r3[i][1]} {r3[i][2]}\n')
        f.write(f'{t[i][0]} {t[i][1]} {t[i][2]}\n')