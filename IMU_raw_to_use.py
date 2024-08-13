## 將原始 IMU 資料轉換成使用格式

import os

# 獲得目前的工作目錄
base_path = os.getcwd()
result_path = os.path.join(base_path, 'IMU_result_data')

# 建立 IMU 文件夾路徑
imu_folder = os.path.join(base_path, 'IMU_raw_data')

# 獲得 IMU 文件夾中所有以 "MT_" 開頭的資料名稱
imu_files = [file for file in os.listdir(imu_folder) if file.startswith("MT_012102F3_030")]

# 讀取 IMU 文件並儲存資料
imu_data = {}
for file_name in imu_files:
    full_path = os.path.join(imu_folder, file_name)
    with open(full_path, 'r') as file:
        lines = file.readlines()
        device_id = None
        for line in lines:
            if line.strip().startswith("//  DeviceId:"):
                device_id = line.split(":")[1].strip()
            elif line.strip().startswith("PacketCounter"):
                    imu_data[device_id] = lines[lines.index(line) + 1:]
            # else:
            #     print("Warning: Device ID not found!")  # 警告沒有找到
# print(imu_data)

# IMU 名稱到映射到骨頭名稱
sensor_to_device_id = {
    'Head': '00B48A6F', 'Sternum': '00B48A50', 'Pelvis': '00B48A71',
    'L_UpArm': '00B48A74', 'R_UpArm': '00B48A4D', 'L_LowArm': '00B48A70', 'R_LowArm': '00B48A5D',
    'L_UpLeg': '00B48A6E', 'R_UpLeg': '00B48A6C', 'L_LowLeg': '00B48A69', 'R_LowLeg': '00B48A73'
}

# 根據設備 ID 映射到 IMU 文件
sensor_to_imu = {}
for sensor_name, device_id in sensor_to_device_id.items():
    for imu_file in imu_files:
        if device_id in imu_file:
            sensor_to_imu[sensor_name] = imu_file
            break
# print(sensor_to_imu)

# 提取並排列需要的資訊及格式
current_time = None
# print(len(imu_data['00B48A4D']))
with open(os.path.join(result_path, 's1_walking2_Xsens_self.sensors'),'wb') as f:
    imu_num_and_frame = f"{len(imu_data)}\t{len(imu_data['00B48A4D'])}\n"
    f.write(imu_num_and_frame.encode())
    for frame in range(len(imu_data['00B48A4D'])):
        # print(frame + 1)
        f.write(str(frame + 1).encode() + b'\n')
        for sensor_name, imu_file in sensor_to_imu.items():
            # print('sensor_name:', sensor_name, 'imu_file:', imu_file)
            device_id = sensor_to_device_id[sensor_name]
            # print('device_id:', device_id)
            imu_lines = imu_data.get(device_id, [])  # 得 IMU 數據，如果不存在則返回空陣列
            # print(device_id, len(imu_lines))
            parts = imu_lines[frame].split()
            time = parts[0]
            quat = '\t'.join(parts[4:8])  # 四元數轉成字串
            accel = '\t'.join(parts[1:4])  # 加速度轉成字串

            # [sensor 1 name]	[IMU local ori quat (w x y z)]	[IMU local accel (x y z)]
            line_to_write = f"{sensor_name}\t{quat}\t{accel}\n"
            f.write(line_to_write.encode())