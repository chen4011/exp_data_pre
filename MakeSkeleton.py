## 製作 datafusion 要用到的骨架，基於 Vicon 骨架進行伸縮

import os
import pandas as pd
import matplotlib.pyplot as plt

# 讀取 .trc 檔案
path = os.getcwd()

# -----記得改輸入的 .trc 檔名-----
data = pd.read_csv('Empty_project_filt_butterworth_0-318.trc', sep='\t')
# -------------------------------

bones = ['Head', 'Sternum', 'Pelvis', 'L_UpArm', 'R_UpArm', 'L_LowArm', 'R_LowArm', 'L_UpLeg', 'R_UpLeg', 'L_LowLeg', 'R_LowLeg']
bone_vicon_skeleton = {'Head':[0,-2.2048216,-12.50419394], 'Sternum':[0.00000254,-3.4628328,-24.2399058], 'Pelvis':[0,1.603502,-9.093962],
        'L_UpArm':[-28.8867596,0,-0.00000508], 'R_UpArm':[28.8867596,0,0.00000508],
        'L_LowArm':[-21.96027866,0,0], 'R_LowArm':[21.96027866,0,0],
        'L_UpLeg':[-0.00000254,0,37.8891038], 'R_UpLeg':[-0.00000254,0,37.8891038],
        'L_LowLeg':[-0.00000254,0,37.54378826], 'R_LowLeg':[-0.00000254,0,37.54378826]}
bone_end = {'Head':[13,12],
            # 'Sternum':[0,0], 'Pelvis':[0,0],
            'L_UpArm':[18,19], 'R_UpArm':[15,16], 'L_LowArm':[19,20], 'R_LowArm':[16,17],
            'L_UpLeg':[6,7], 'R_UpLeg':[0,1], 'L_LowLeg':[7,8], 'R_LowLeg':[1,2]}

# 分類 x, y, z，並將 DataFrame 轉換為數值型，取平均值*100，並計算最大最小值的差
data_x = data.iloc[:, 2::3].astype(float)
data_y = data.iloc[:, 3::3].astype(float)
data_z = data.iloc[:, 4::3].astype(float)
# print(data_x)

ave_bone_length = {}

for bone in bones:
    sum_bone_length = 0
    for i in range(len(data)):
        if bone == 'Sternum' or bone == 'Pelvis':
            mid_shoulder_x = (data_x.iloc[i][15] + data_x.iloc[i][18])/2
            mid_shoulder_y = (data_y.iloc[i][15] + data_y.iloc[i][18])/2
            mid_shoulder_z = (data_z.iloc[i][15] + data_z.iloc[i][18])/2
            mid_hip_x = (data_x.iloc[i][0] + data_x.iloc[i][6])/2
            mid_hip_y = (data_y.iloc[i][0] + data_y.iloc[i][6])/2
            mid_hip_z = (data_z.iloc[i][0] + data_z.iloc[i][6])/2
            if bone == 'Sternum':
                body_length = ((mid_shoulder_x - data_x.iloc[i][12])**2 + (mid_shoulder_y - data_y.iloc[i][12])**2
                            + (mid_shoulder_z - data_z.iloc[i][12])**2)**0.5*100
            elif bone == 'Pelvis':
                body_length = ((mid_shoulder_x - mid_hip_x)**2 + (mid_shoulder_y - mid_hip_y)**2
                                + (mid_shoulder_z - mid_hip_z)**2)**0.5*100
        else:
            bone_length = ((data_x.iloc[i][bone_end[bone][0]] - data_x.iloc[i][bone_end[bone][1]])**2
                        + (data_y.iloc[i][bone_end[bone][0]] - data_y.iloc[i][bone_end[bone][1]])**2
                        + (data_z.iloc[i][bone_end[bone][0]] - data_z.iloc[i][bone_end[bone][1]])**2)**0.5*100
        sum_bone_length = sum_bone_length + bone_length
    ave_bone_length[bone] = sum_bone_length/len(data)
print(ave_bone_length)
# ave_x = data_x.mean()
# ave_y = data_y.mean()
# ave_z = data_z.mean()
# # print(ave_y)

bone_x = []
bone_y = []
bone_z = []
bone_info = {}
for bone in bones:
    vicon_length = ((bone_vicon_skeleton[bone][0])**2 + (bone_vicon_skeleton[bone][1])**2 + (bone_vicon_skeleton[bone][2])**2)**0.5
    print(vicon_length)
    if bone == 'Head' or bone == 'Pelvis':
        print(bone)
        bone_vec = [bone_vicon_skeleton[bone][0], bone_vicon_skeleton[bone][1], bone_vicon_skeleton[bone][2]]
    # else:
    #     body_length = ((ave_x.iloc[bone_end[bone][0]] - ave_x.iloc[bone_end[bone][1]])**2 
    #                    + (ave_y.iloc[bone_end[bone][0]] - ave_y.iloc[bone_end[bone][1]])**2 
    #                    + (ave_z.iloc[bone_end[bone][0]] - ave_z.iloc[bone_end[bone][1]])**2)**0.5
    else:
        bone_vec = [bone_vicon_skeleton[bone][0]*ave_bone_length[bone]/vicon_length,
                    bone_vicon_skeleton[bone][1]*ave_bone_length[bone]/vicon_length,
                    bone_vicon_skeleton[bone][2]*ave_bone_length[bone]/vicon_length]
    bone_x.append(bone_vec[0])
    bone_y.append(bone_vec[1])
    bone_z.append(bone_vec[2])
    bone_info[bone] = bone_vec
with open(os.path.join(path,'bone_info_self.txt'), 'w+') as f:
    f.write(str(bone_info))
# print(bone_y)
# print(bone_z)

dis_x = max(bone_x) - min(bone_x)
dis_y = max(bone_y) - min(bone_y)
dis_z = max(bone_z) - min(bone_z)
# print(dis_x, dis_y, dis_z)

# 將 data 分成五組，[ 6 個部位的 x 值, 6 個部位的 y 值, 6 個部位的 z 值]
right_leg = [[bone_x[0], bone_x[0]+bone_x[8], bone_x[0]+bone_x[8]+bone_x[10]], [bone_y[0], bone_y[0]+bone_y[8], bone_y[0]+bone_y[8]+bone_y[10]], [bone_z[0], bone_z[0]+bone_z[8], bone_z[0]+bone_z[8]+bone_z[10]]]
left_leg = [[bone_x[0], bone_x[0]+bone_x[7], bone_x[0]+bone_x[7]+bone_x[9]], [bone_y[0], bone_y[0]+bone_y[7], bone_y[0]+bone_y[7]+bone_y[9]], [bone_z[0], bone_z[0]+bone_z[7], bone_z[0]+bone_z[7]+bone_z[9]]]
body = [[bone_x[0], bone_x[0]+bone_x[1], bone_x[0]+bone_x[1]+bone_x[2]], [bone_y[0], bone_y[0]+bone_y[1], bone_y[0]+bone_y[1]+bone_y[2]], [bone_z[0], bone_z[0]+bone_z[1], bone_z[0]+bone_z[1]+bone_z[2]]]
right_arm = [[bone_x[0]+bone_x[1], bone_x[0]+bone_x[1]+bone_x[4], bone_x[0]+bone_x[1]+bone_x[4]+bone_x[6]], [bone_y[0]+bone_y[1], bone_y[0]+bone_y[1]+bone_y[4], bone_y[0]+bone_y[1]+bone_y[4]+bone_y[6]], [bone_z[0]+bone_z[1], bone_z[0]+bone_z[1]+bone_z[4], bone_z[0]+bone_z[1]+bone_z[4]+bone_z[6]]]
left_arm = [[bone_x[0]+bone_x[1], bone_x[0]+bone_x[1]+bone_x[3], bone_x[0]+bone_x[1]+bone_x[3]+bone_x[5]], [bone_y[0]+bone_y[1], bone_y[0]+bone_y[1]+bone_y[3], bone_y[0]+bone_y[1]+bone_y[3]+bone_y[5]], [bone_z[0]+bone_z[1], bone_z[0]+bone_z[1]+bone_z[3], bone_z[0]+bone_z[1]+bone_z[3]+bone_z[5]]]
# print(left_leg)
# # 將 data 分成五組，[ 6 個部位的 x 值, 6 個部位的 y 值, 6 個部位的 z 值]
# right_leg = [[bone_x['R_UpLeg'], bone_x['R_UpLeg']+bone_x['R_LowLeg']], [bone_y['R_UpLeg'], bone_y['R_UpLeg']+bone_y['R_LowLeg']], [bone_z['R_UpLeg'], bone_z['R_UpLeg']+bone_z['R_LowLeg']]]
# left_leg = [[bone_x['L_UpLeg'], bone_x['L_UpLeg']+bone_x['L_LowLeg']], [bone_y['L_UpLeg'], bone_y['L_UpLeg']+bone_y['L_LowLeg']], [bone_z['L_UpLeg'], bone_z['L_UpLeg']+bone_z['L_LowLeg']]]
# body = [[bone_x['Head'], bone_x['Head']+bone_x['Sternum'], bone_x['Head']+bone_x['Sternum']+bone_x['Pelvis']], [bone_y['Head'], bone_y['Head']+bone_y['Sternum'], bone_y['Head']+bone_y['Sternum']+bone_y['Pelvis']], [bone_z['Head'], bone_z['Head']+bone_z['Sternum'], bone_z['Head']+bone_z['Sternum']+bone_z['Pelvis']]]
# right_arm = [[bone_x['R_UpArm'], bone_x['R_UpArm']+bone_x[6]], [bone_y['R_UpArm'], bone_y['R_UpArm']+bone_y[6]], [bone_z['R_UpArm'], bone_z['R_UpArm']+bone_z[6]]]
# left_arm = [[bone_x['L_UpArm'], bone_x['L_UpArm']+bone_x[5]], [bone_y['L_UpArm'], bone_y['L_UpArm']+bone_y[5]], [bone_z['L_UpArm'], bone_z['L_UpArm']+bone_z[5]]]
# # print(left_leg)

fig = plt.figure(figsize=(6,6))
pos = fig.add_subplot(111, projection='3d')   # 設定為 3D 圖表
pos.set_title('3D position',fontsize=20)      # 設定圖表 title
pos.set_xlabel('x',fontdict={'size':15},labelpad=8, color='#f00')  # 設定 x 軸標題
pos.set_ylabel('y',fontdict={'size':15},labelpad=8, color='#0f0')  # 設定 y 軸標題
pos.set_zlabel('z',fontdict={'size':15},labelpad=8, color='#00f')  # 設定 z 軸標題
pos.tick_params(axis='x', labelsize=12, labelcolor='#f00' )  # 設定 x 軸標籤文字
pos.tick_params(axis='y', labelsize=12, labelcolor='#0f0' )  # 設定 y 軸標籤文字
pos.tick_params(axis='z', labelsize=12, labelcolor='#00f' )  # 設定 z 軸標籤文字

# 繪製 3D 座標點
pos.scatter(right_leg[0], right_leg[1], right_leg[2], color='red', marker='o', label='Right Leg')
pos.plot(right_leg[0], right_leg[1], right_leg[2], color='r', label='Right Leg')
pos.scatter(left_leg[0], left_leg[1], left_leg[2], color='orange', marker='o', label='Left Leg')
pos.plot(left_leg[0], left_leg[1], left_leg[2], color='orange', label='Left Leg')
pos.scatter(body[0], body[1], body[2], color='green', marker='x', label='Body')
pos.plot(body[0], body[1], body[2], color='g', label='Body')
pos.scatter(right_arm[0], right_arm[1], right_arm[2], color='blue', marker='s', label='Right arm')
pos.plot(right_arm[0], right_arm[1], right_arm[2], color='b', label='Right arm')
pos.scatter(left_arm[0], left_arm[1], left_arm[2], color='purple', marker='s', label='Left arm')
pos.plot(left_arm[0], left_arm[1], left_arm[2], color='purple', label='Left arm')

# 顯示圖例
pos.legend()

# 顯示圖形
plt.gca().set_box_aspect([dis_x, dis_y, dis_z])  # 設定 x, y, z 軸比例相同
# plt.gca().set_box_aspect([16,1,36])  # 設定 x, y, z 軸比例相同
plt.show()