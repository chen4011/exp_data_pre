## 畫出當前的 3D 姿勢
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 建立 pose_result 資料夾
if not os.path.exists('pose_result'):
    os.makedirs('pose_result')

# 獲取文件名和擴展名
filename, ext = os.path.splitext('3dpose_prediction_triangulation.txt')

# 如果擴展名是 .trc，則讀取文件
if ext == '.trc':
    fig = plt.figure(figsize=(6,6))
    pos = fig.add_subplot(111, projection='3d')   # 設定為 3D 圖表
    pos.set_title('3D position',fontsize=20)      # 設定圖表 title
    pos.set_xlabel('x',fontdict={'size':15},labelpad=8, color='#f00')  # 設定 x 軸標題
    pos.set_ylabel('y',fontdict={'size':15},labelpad=8, color='#0f0')  # 設定 y 軸標題
    pos.set_zlabel('z',fontdict={'size':15},labelpad=8, color='#00f')  # 設定 z 軸標題
    pos.tick_params(axis='x', labelsize=12, labelcolor='#f00' )  # 設定 x 軸標籤文字
    pos.tick_params(axis='y', labelsize=12, labelcolor='#0f0' )  # 設定 y 軸標籤文字
    pos.tick_params(axis='z', labelsize=12, labelcolor='#00f' )  # 設定 z 軸標籤文字

    # 讀取 .trc 檔案
    data = pd.read_csv('Empty_project_filt_butterworth_0-726.trc', sep='\t')

    # 分類 x, y, z，並將 DataFrame 轉換為數值型，取平均值*100，並計算最大最小值的差
    data_x = data.iloc[:, 2::3].astype(float)
    data_y = data.iloc[:, 3::3].astype(float)
    data_z = data.iloc[:, 4::3].astype(float)
    # print(data_y)
    ave_x = data_x.mean()/10
    ave_y = data_y.mean()/10
    ave_z = data_z.mean()/10
    # print(ave_y)
    dis_x = ave_x.max() - ave_x.min()
    dis_y = ave_y.max() - ave_y.min()
    dis_z = ave_z.max() - ave_z.min()
    # print(dis_x, dis_y, dis_z)

    # 將 data 分成五組，[ 6 個部位的 x 值, 6 個部位的 y 值, 6 個部位的 z 值]
    right_leg = [pd.Series(ave_x.iloc[0:6]), pd.Series(ave_y.iloc[0:6]), pd.Series(ave_z.iloc[0:6])]
    left_leg = [pd.Series(ave_x.iloc[6:12]), pd.Series(ave_y.iloc[6:12]), pd.Series(ave_z.iloc[6:12])]
    body = [pd.Series(ave_x.iloc[12:15]), pd.Series(ave_y.iloc[12:15]), pd.Series(ave_z.iloc[12:15])]
    right_arm = [pd.Series(ave_x.iloc[15:18]), pd.Series(ave_y.iloc[15:18]), pd.Series(ave_z.iloc[15:18])]
    left_arm = [pd.Series(ave_x.iloc[18:21]), pd.Series(ave_y.iloc[18:21]), pd.Series(ave_z.iloc[18:21])]
    # print(left_leg)
    # 繪製 3D 座標點
    pos.scatter(right_leg[0].values, right_leg[1].values, right_leg[2].values, color='red', marker='o', label='Right Leg')
    pos.plot(right_leg[0].values, right_leg[1].values, right_leg[2].values, color='r', label='Right Leg')
    pos.scatter(left_leg[0].values, left_leg[1].values, left_leg[2].values, color='orange', marker='o', label='Left Leg')
    pos.plot(left_leg[0].values, left_leg[1].values, left_leg[2].values, color='orange', label='Left Leg')
    pos.scatter(body[0].values, body[1].values, body[2].values, color='green', marker='x', label='Body')
    pos.plot(body[0].values, body[1].values, body[2].values, color='g', label='Body')
    pos.scatter(right_arm[0].values, right_arm[1].values, right_arm[2].values, color='blue', marker='s', label='Right arm')
    pos.plot(right_arm[0].values, right_arm[1].values, right_arm[2].values, color='b', label='Right arm')
    pos.scatter(left_arm[0].values, left_arm[1].values, left_arm[2].values, color='purple', marker='s', label='Left arm')
    pos.plot(left_arm[0].values, left_arm[1].values, left_arm[2].values, color='purple', label='Left arm')

    # 顯示圖例
    pos.legend()

    # 顯示圖形
    # plt.gca().set_box_aspect([dis_x, dis_y, dis_z])  # 設定 x, y, z 軸比例相同
    # plt.gca().set_box_aspect([16,1,36])  # 設定 x, y, z 軸比例相同
    ax = plt.gca()
    # pos.set_xlim([ave_x.min(), ave_x.max()])  # 设置 x 轴范围
    # pos.set_ylim([ave_y.min(), ave_y.max()])  # 设置 y 轴范围
    # pos.set_zlim([ave_x.min(), ave_z.max()])  # 设置 z 轴范围

    # pos.set_xticks(np.arange(ave_x.min(), ave_x.max(), 20))  # 设置 x 轴刻度
    # pos.set_yticks(np.arange(ave_y.min(), ave_y.max(), 20))  # 设置 y 轴刻度
    # pos.set_zticks(np.arange(ave_x.min(), ave_z.max(), 20))  # 设置 z 轴刻度

    # 繪製 x 軸
    pos.quiver(0, 0, 0, 1, 0, 0, color='r', length=50, arrow_length_ratio=0.1)
    # 繪製 y 軸
    pos.quiver(0, 0, 0, 0, 1, 0, color='g', length=50, arrow_length_ratio=0.1)
    # 繪製 z 軸
    pos.quiver(0, 0, 0, 0, 0, 1, color='b', length=50, arrow_length_ratio=0.1)

    pos.set_xlim([-100, 100])  # 设置 x 轴范围
    pos.set_ylim([-100, 100])  # 设置 y 轴范围
    pos.set_zlim([-100, 100])  # 设置 z 轴范围

    pos.set_xticks(np.arange(-100, 100, 20))  # 设置 x 轴刻度
    pos.set_yticks(np.arange(-100, 100, 20))  # 设置 y 轴刻度
    pos.set_zticks(np.arange(-100, 100, 20))  # 设置 z 轴刻度

    pos.set_box_aspect([200, 200, 200])  # 设置 x, y, z 轴比例相同
    plt.show()

# 如果擴展名是 .txt，則繼續
elif ext == '.txt':
    # 讀取文件的每一行
    with open(filename + ext, 'r') as f:
        lines = f.readlines()
        # print(type(lines))
    # 清理非數字字符並轉換為 numpy 數組
    # arrays = [np.fromstring(line.replace('[', '').replace(']', ''), sep=' ')/10 for line in lines]
    with open('limb_length.txt', 'w') as f:
        f.write('ru_leg_length rl_leg_length lu_leg_length ll_leg_length lu_arm_length ll_arm_length ru_arm_length rl_arm_length\n')
        per = 1
        print(len(lines))
        for i in range(0, len(lines)):
            if i % per == 0:
                line = lines[i]
                # 移除外部的方括号并分割字符串
                data_str = ''.join(line).strip('[]')
                data_list = data_str.split('] [')

                # 将每个子字符串转换为浮点数数组
                arrays = [np.fromstring(item, sep=' ') for item in data_list]

                # 转换为 NumPy 数组
                arrays_np = np.array(arrays)/10

                # 定义 x 轴旋转 90 度的旋转矩阵
                R_x_90 = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

                # 应用旋转矩阵
                rotated_coordinates = np.dot(R_x_90, arrays_np.T).T
                # print(rotated_coordinates)

                # rotated_coordinates 现在包含旋转后的坐标数据

                # # 將每一行的數組中的第1、4、7、10...45個元素分成一組
                # x = [array[np.arange(0, 46, 3)] for array in arrays]
                # # 將每一行的數組中的第2、5、8、11...46個元素分成一組
                # y = [array[np.arange(1, 47, 3)] for array in arrays]
                # # 將每一行的數組中的第3、6、9、12...47個元素分成一組
                # z = [array[np.arange(2, 48, 3)] for array in arrays]

                # 分配到 x, y, z
                x = rotated_coordinates[:, 0]
                y = rotated_coordinates[:, 1]
                z = rotated_coordinates[:, 2]

                fig = plt.figure(figsize=(6,6))
                pos = fig.add_subplot(111, projection='3d')   # 設定為 3D 圖表
                pos.set_title('3D position',fontsize=20)      # 設定圖表 title
                # pos.set_xlabel('x',fontdict={'size':15},labelpad=8, color='#f00')  # 設定 x 軸標題
                # pos.set_ylabel('y',fontdict={'size':15},labelpad=8, color='#0f0')  # 設定 y 軸標題
                # pos.set_zlabel('z',fontdict={'size':15},labelpad=8, color='#00f')  # 設定 z 軸標題
                # pos.tick_params(axis='x', labelsize=12, labelcolor='#f00' )  # 設定 x 軸標籤文字
                # pos.tick_params(axis='y', labelsize=12, labelcolor='#0f0' )  # 設定 y 軸標籤文字
                # pos.tick_params(axis='z', labelsize=12, labelcolor='#00f' )  # 設定 z 軸標籤文字

                root = [x[0], y[0], z[0]]
                # root = [0,0,0]
                body = [[x[0]-root[0], x[7]-root[0], x[8]-root[0], x[9]-root[0]],
                        [y[0]-root[1], y[7]-root[1], y[8]-root[1], y[9]-root[1]],
                        [z[0]-root[2], z[7]-root[2], z[8]-root[2], z[9]-root[2]]]
                right_leg = [[x[1]-root[0],x[2]-root[0],x[3]-root[0]],
                            [y[1]-root[1],y[2]-root[1],y[3]-root[1]],
                            [z[1]-root[2],z[2]-root[2],z[3]-root[2]]]
                left_leg = [[x[4]-root[0],x[5]-root[0],x[6]-root[0]],
                            [y[4]-root[1],y[5]-root[1],y[6]-root[1]],
                            [z[4]-root[2],z[5]-root[2],z[6]-root[2]]]
                left_arm = [[x[10]-root[0],x[11]-root[0],x[12]-root[0]],
                            [y[10]-root[1],y[11]-root[1],y[12]-root[1]],
                            [z[10]-root[2],z[11]-root[2],z[12]-root[2]]]
                right_arm = [[x[13]-root[0],x[14]-root[0],x[15]-root[0]],
                            [y[13]-root[1],y[14]-root[1],y[15]-root[1]],
                            [z[13]-root[2],z[14]-root[2],z[15]-root[2]]]
                
                ru_leg_length = ((x[1]-x[2])**2 + (y[1]-y[2])**2 + (z[1]-z[2])**2)**0.5
                rl_leg_length = ((x[2]-x[3])**2 + (y[2]-y[3])**2 + (z[2]-z[3])**2)**0.5
                lu_leg_length = ((x[4]-x[5])**2 + (y[4]-y[5])**2 + (z[4]-z[5])**2)**0.5
                ll_leg_length = ((x[5]-x[6])**2 + (y[5]-y[6])**2 + (z[5]-z[6])**2)**0.5
                lu_arm_length = ((x[10]-x[11])**2 + (y[10]-y[11])**2 + (z[10]-z[11])**2)**0.5
                ll_arm_length = ((x[11]-x[12])**2 + (y[11]-y[12])**2 + (z[11]-z[12])**2)**0.5
                ru_arm_length = ((x[13]-x[14])**2 + (y[13]-y[14])**2 + (z[13]-z[14])**2)**0.5
                rl_arm_length = ((x[14]-x[15])**2 + (y[14]-y[15])**2 + (z[14]-z[15])**2)**0.5
                f.write(f'{ru_leg_length} {rl_leg_length} {lu_leg_length} {ll_leg_length} {lu_arm_length} {ll_arm_length} {ru_arm_length} {rl_arm_length}\n')

                # pos.scatter(body[0], body[1], body[2], color='green', marker='x', label='Body')
                # pos.scatter(right_leg[0], right_leg[1], right_leg[2], color='red', marker='o', label='Right Leg')
                # pos.scatter(left_leg[0], left_leg[1], left_leg[2], color='orange', marker='o', label='Left Leg')
                # pos.scatter(right_arm[0], right_arm[1], right_arm[2], color='blue', marker='s', label='Right arm')
                # pos.scatter(left_arm[0], left_arm[1], left_arm[2], color='purple', marker='s', label='Left arm')
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
                # plt.gca().set_box_aspect([dis_x, dis_y, dis_z])  # 設定 x, y, z 軸比例相同
                # plt.gca().set_box_aspect([16,1,36])  # 設定 x, y, z 軸比例相同
                pos = plt.gca()
                # pos.set_xlim([ave_x.min(), ave_x.max()])  # 设置 x 轴范围
                # pos.set_ylim([ave_y.min(), ave_y.max()])  # 设置 y 轴范围
                # pos.set_zlim([ave_x.min(), ave_z.max()])  # 设置 z 轴范围

                # pos.set_xticks(np.arange(ave_x.min(), ave_x.max(), 20))  # 设置 x 轴刻度
                # pos.set_yticks(np.arange(ave_y.min(), ave_y.max(), 20))  # 设置 y 轴刻度
                # pos.set_zticks(np.arange(ave_x.min(), ave_z.max(), 20))  # 设置 z 轴刻度

                # # 繪製 x 軸
                # pos.quiver(0, 0, 0, 1, 0, 0, color='r', length=50, arrow_length_ratio=0.1)
                # # 繪製 y 軸
                # pos.quiver(0, 0, 0, 0, 1, 0, color='g', length=50, arrow_length_ratio=0.1)
                # # 繪製 z 軸
                # pos.quiver(0, 0, 0, 0, 0, 1, color='b', length=50, arrow_length_ratio=0.1)

                pos.set_xlim([-100, 100])  # 设置 x 轴范围
                pos.set_ylim([-100, 100])  # 设置 y 轴范围
                pos.set_zlim([-100, 100])  # 设置 z 轴范围

                pos.set_xticks(np.arange(-100, 100, 20))  # 设置 x 轴刻度
                pos.set_yticks(np.arange(-100, 100, 20))  # 设置 y 轴刻度
                pos.set_zticks(np.arange(-100, 100, 20))  # 设置 z 轴刻度

                pos.set_box_aspect([200, 200, 200])  # 设置 x, y, z 轴比例相同
                # pos.view_init(elev=90, azim=-90) # 設定視角, elve:上下旋轉, azim:左右旋轉
                # plt.show()

                # 保存圖像
                plt.savefig(f'pose_result/pose_{(i/per)+1}.png')
                # print(f'pose_{i+1}.png saved')

                # 清理當前圖形，以便下一次循环可以开始一个新的图形
                plt.clf()
                plt.close()  # 關閉圖形窗口以釋放內存
                print(f'pose_{(i/per)+1} cleared')