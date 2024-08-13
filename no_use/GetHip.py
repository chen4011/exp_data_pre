# import pandas as pd
# import json

# # 讀取 .trc 檔案
# data = pd.read_csv('Empty_project_filt_butterworth_0-3188.trc', sep='\t')

# # 分類 x, y, z，並將 DataFrame 轉換為數值型，取平均值*100，並計算最大最小值的差
# right_hip = data.iloc[:, 2:5].astype(float)
# left_hip = data.iloc[:, 20:23].astype(float)
# hip_list = []
# for i in range(0, len(data)):
#     hip_x = (right_hip.iloc[i, 0] + left_hip.iloc[i, 0])/2
#     hip_y = (right_hip.iloc[i, 1] + left_hip.iloc[i, 1])/2
#     hip_z = (right_hip.iloc[i, 2] + left_hip.iloc[i, 2])/2
#     hip_dict = {'Hips': [hip_x, hip_y, hip_z]}
#     hip_list.append(hip_dict)
# with open ('gt_skel_gbl_pos_hip.txt', 'w') as f:
#     for hip_dict in hip_list:
#         f.write(json.dumps(hip_dict) + '\n')