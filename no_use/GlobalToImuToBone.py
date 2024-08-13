## 計算 bone to imu local to imu global to unknown global

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import numpy as np


def rotation_matrix_to_euler_angles(R) :
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

# def quaternion_to_euler_angles(quaternion) :
#     r = R.from_quat(quaternion)
#     euler = r.as_euler('zyx', degrees=True)
#     return euler

bones = [
    # 'Head',
    'Sternum','Pelvis', 'L_UpArm', 'R_UpArm', 'L_LowArm', 'R_LowArm', 'L_UpLeg', 'R_UpLeg', 'L_LowLeg', 'R_LowLeg']
# [w, x, y, z]
# calib_imu_bone = {
#     'Head': [-0.101626, -0.601006, -0.781181, 0.134984], 'Sternum': [0.389136, 0.563679, 0.639287, -0.349501],
#     'Pelvis': [-0.404335, 0.583153, -0.577675, -0.403407],
#     'L_UpArm': [0.330458, -0.937681, -0.0948915, 0.0504626], 'R_UpArm': [0.104461, -0.0658098, 0.941085, 0.314827],
#     'L_LowArm': [-0.0421835, -0.0544204, 0.971188, -0.228149], 'R_LowArm': [0.0475826, -0.00458409, 0.992402, -0.113369],
#     'L_UpLeg': [0.751715, 0.130601, -0.645948, 0.024886], 'R_UpLeg': [0.238334, 0.618576, -0.257819, 0.702915],
#     'L_LowLeg': [0.600816, 0.434819, 0.580591, -0.335956], 'R_LowLeg': [-0.102554, -0.622879, 0.202039, -0.748789],
#     'L_Foot': [-0.435842, -0.52429, 0.731421, 0.0135762], 'R_Foot': [-0.482439, -0.543732, -0.671906, -0.141955]}
calib_imu_bone = {
    'Sternum': [0.451785, -0.028639000000000137, 0.890597, 0.043668], 'Pelvis': [0.09951199999999988, -0.804928, 0.15428499999999992, 0.564256], 'L_UpArm': [-0.028149999999999935, 0.4901779999999999, 0.8221760000000001, -0.28802600000000006], 'R_UpArm': [-0.1713200000000001, -0.6883820000000002, 0.5656129999999999, 0.4205479999999999], 'L_LowArm': [0.13148100000000004, 0.6442179999999998, 0.7412480000000001, -0.13508200000000004], 'R_LowArm': [-0.11835599999999995, 0.7254920000000001, -0.6674799999999999, -0.11884399999999998], 'L_UpLeg': [-0.660384, 0.2920460000000001, -0.580802, -0.37586], 'R_UpLeg': [0.23099800000000012, 0.586511, 0.21283900000000003, -0.746555], 'L_LowLeg': [0.724795, 0.20461099999999985, 0.629194, -0.19214699999999996], 'R_LowLeg': [0.26815099999999986, -0.690254, 0.19148799999999994, 0.644187]}

# [w, x, y, z]
calib_imu_ref = {
    'Head': [0.70710678, -0.70710678, 0.0, 0.0], 'Sternum': [0.70710678, -0.70710678, 0.0, 0.0],
    'Pelvis': [0.70710678, -0.70710678, 0.0, 0.0],
    'L_UpArm': [0.70710678, -0.70710678, 0.0, 0.0], 'R_UpArm': [0.70710678, -0.70710678, 0.0, 0.0],
    'L_LowArm': [0.70710678, -0.70710678, 0.0, 0.0], 'R_LowArm': [0.70710678, -0.70710678, 0.0, 0.0],
    'L_UpLeg': [0.70710678, -0.70710678, 0.0, 0.0], 'R_UpLeg': [0.70710678, -0.70710678, 0.0, 0.0],
    'L_LowLeg': [0.70710678, -0.70710678, 0.0, 0.0], 'R_LowLeg': [0.70710678, -0.70710678, 0.0, 0.0],
    'L_Foot': [0.70710678, -0.70710678, 0.0, 0.0], 'R_Foot': [0.70710678, -0.70710678, 0.0, 0.0]}
# calib_imu_ref = {
#     'Head': [0.706433, -0.707768, 0.00400315, 0.000770449], 'Sternum': [0.707303, -0.706412, 0.0194452, 0.0180635],
#     'Pelvis': [0.706467, -0.703579, -0.0543374, -0.0541108],
#     'L_UpArm': [-0.708866, 0.705206, 0.00635185, 0.0124047], 'R_UpArm': [0.703752, -0.706316, 0.0508097, 0.057186],
#     'L_LowArm': [0.70641, -0.707376, -0.0204765, -0.0136335], 'R_LowArm': [0.708595, -0.704426, 0.030979, 0.0267902],
#     'L_UpLeg': [0.705046, -0.707543, -0.0326425, -0.035041], 'R_UpLeg': [0.70881, -0.704532, 0.0226213, 0.0266781],
#     'L_LowLeg': [0.701376, -0.704415, -0.074554, -0.0794547], 'R_LowLeg': [0.703044, -0.703478, -0.0748645, -0.0724074],
#     'L_Foot': [0.699219, -0.701063, -0.100102, -0.0978899], 'R_Foot': [0.702827, -0.704787, -0.0702866, -0.0660937]}
# s2_walking1 = [x, y, z, w]
# calib_imu_ref = {
#     'Head':[-0.708288, -0.00108211, 0.000997401, 0.705922], 'Sternum':[-0.707172, 0.0303025, 0.0329436, 0.705624],
#     'Pelvis':[-0.704272, -0.0573664, -0.0536933, 0.705569],
#     'L_UpArm':[-0.706563, -0.010471, -0.00901061, 0.707515], 'R_UpArm':[-0.705996, 0.0334204, 0.0375802, 0.706428],
#     'L_LowArm':[-0.705321, -0.0304879, -0.0297323, 0.707608], 'R_LowArm':[-0.705783, 0.0284825, 0.0277421, 0.707312],
#     'L_UpLeg':[-0.707176, -0.0169823, -0.0156674, 0.70666], 'R_UpLeg':[-0.704533, 0.0373549, 0.0398458, 0.707566],
#     'L_LowLeg':[-0.699372, -0.104172, -0.105608, 0.699195], 'R_LowLeg':[-0.700178, -0.100213, -0.0981182, 0.700058],
#     'L_Foot':[-0.697881, -0.119846, -0.117775, 0.696224], 'R_Foot':[-0.69013, -0.163434, -0.158986, 0.686828]}

# inch
bone_info = {
    'Head': ('Neck', 'Head', (0.0, -0.86804, -4.922911)), 'Sternum': ('Spine3', 'Neck', (1e-06, -1.36332, -9.54327)),
    'Pelvis': ('Spine', 'Spine1', (0.0, 0.6313, -3.5803)),
    'L_UpArm': ('LeftArm', 'LeftForeArm', (-11.37274, 0.0, -2e-06)), 'R_UpArm': ('RightArm', 'RightForeArm', (11.37274, 0.0, 2e-06)),
    'L_LowArm': ('LeftForeArm', 'LeftHand', (-8.645779, 0.0, 0.0)),'R_LowArm': ('RightForeArm', 'RightHand', (8.645779, 0.0, 0.0)),
    'L_UpLeg': ('LeftUpLeg', 'LeftLeg', (-1e-06, 0.0, 14.91697)), 'R_UpLeg': ('RightUpLeg', 'RightLeg', (-1e-06, 0.0, 14.91697)),
    'L_LowLeg': ('LeftLeg', 'LeftFoot', (-1e-06, 0.0, 14.781019)), 'R_LowLeg': ('RightLeg', 'RightFoot', (-1e-06, 0.0, 14.781019)),
    'L_Foot': ('LeftFoot', 'LeftToeBase', (-0.0, -5.67053, 2.063911)), 'R_Foot': ('RightFoot', 'RightToeBase', (-0.0, -5.67053, 2.063911))}
# [w, x, y, z]
#11
# imu_data = {
#     'Sternum':((0.722664, 0.265812, -0.557195, 0.310860),(5.244144, 5.008168, 7.579444)),
#     'Pelvis':((0.810503, 0.104069, -0.558526, 0.142488),(9.988634, 0.436752, 2.116104)),
#     'L_UpArm':((0.647013, -0.639421, 0.145380, -0.389075),(7.569812, -57.124695, -20.411489)),
#     'R_UpArm':((-0.197248, -0.397139, -0.479471, -0.7572861),(-20.964314, 55.020579, -0.518834)),
#     'L_LowArm':((0.446149, -0.852413, 0.099120, -0.254005),(79.761893, 34.168866, 21.163545)),
#     'R_LowArm':((-0.084626, -0.511374, -0.646478, -0.559823),(117.639088, -41.845026, 28.463330)),
#     'L_UpLeg':((-0.216307, -0.684033, 0.225137, -0.659259),(9.854170, -0.320752, -0.807164)),
#     'R_UpLeg':((0.629350, -0.236486, -0.681216, -0.289721),(9.181222, 0.118005, -2.328239)),
#     'L_LowLeg':((-0.117654, -0.735773, 0.126332, -0.654856),(9.750855, -0.453599, -0.016821)),
#     'R_LowLeg':((0.654956, -0.036416, -0.754003, -0.034441),(9.502828, 1.447133, -0.854224))}
# 12
# imu_data = {
#     'Sternum':((0.722664, 0.265812, -0.557195, 0.310860),(5.244144, 5.008168, 7.579444)),
#     'Pelvis':((0.675247, -0.357217, -0.455393, -0.457225),(9.988634, 0.436752, 2.116104)),
#     'L_UpArm':((0.070849, -0.291458, 0.562764, -0.770279),(7.569812, -57.124695, -20.411489)),
#     'R_UpArm':((-0.663202, -0.506657, -0.077899, -0.545338),(-20.964314, 55.020579, -0.518834)),
#     'L_LowArm':((0.234556, -0.557647, 0.582243, -0.543145),(79.761893, 34.168866, 21.163545)),
#     'R_LowArm':((0.529949, 0.679059, 0.220795, 0.457473),(117.639088, -41.845026, 28.463330)),
#     'L_UpLeg':((-0.667328, -0.260901, 0.638915, -0.279984),(9.854170, -0.320752, -0.807164)),
#     'R_UpLeg':((-0.328878, 0.638649, 0.329868, 0.612498),(9.181222, 0.118005, -2.328239)),
#     'L_LowLeg':((-0.653599, -0.238195, 0.700948, -0.157302),(9.750855, -0.453599, -0.016821)),
#     'R_LowLeg':((0.407533, -0.622234, -0.471917, -0.473324),(9.502828, 1.447133, -0.854224))}
#13
# imu_data = {
#     'Sternum':((0.722664, 0.265812, -0.557195, 0.310860),(5.244144, 5.008168, 7.579444)),
#     'Pelvis':((0.519851, 0.379287, -0.342813, 0.684380),(9.988634, 0.436752, 2.116104)),
#     'L_UpArm':((0.765940, -0.571954, -0.263968, -0.128549),(7.569812, -57.124695, -20.411489)),
#     'R_UpArm':((0.162707, -0.200907, -0.441088, -0.859421),(-20.964314, 55.020579, -0.518834)),
#     'L_LowArm':((0.587297, -0.685040, -0.429509, -0.036387),(79.761893, 34.168866, 21.163545)),
#     'R_LowArm':((0.117782, -0.271433, -0.670474, -0.680379),(117.639088, -41.845026, 28.463330)),
#     'L_UpLeg':((0.102904, -0.647777, -0.158522, -0.738015),(9.854170, -0.320752, -0.807164)),
#     'R_UpLeg':((0.668971, 0.068483, -0.732855, 0.103498),(9.181222, 0.118005, -2.328239)),
#     'L_LowLeg':((0.096790, -0.722379, -0.172210, -0.662679),(9.750855, -0.453599, -0.016821)),
#     'R_LowLeg':((0.347808, 0.583535, -0.527589, 0.510065),(9.502828, 1.447133, -0.854224))}
#15
imu_data = {
    'Head':((0.899248, -0.133849, -0.182418, -0.374381),( 4.906988, -2.134717, 9.616141)),
    'Sternum':((-0.001576, 0.489307, 0.037125, 0.871320),(11.283996, 1.426187, 1.841397)),
    'Pelvis':((0.800237, 0.104863, -0.567135, 0.164261),(9.176794, 0.106474, 5.052990)),
    'L_UpArm':((0.002746, 0.385401, 0.373083, 0.843959),(11.965171, 2.372802, 2.247465)),
    'R_UpArm':((0.243629, 0.380928, -0.403916, 0.795230),(11.467056, -3.442773, 3.605060)),
    'L_LowArm':((0.622051, -0.105769, 0.443719, 0.636379),(-13.443395, -3.643662, 5.580413)),
    'R_LowArm':((0.481626, 0.104373, 0.342114, -0.800063),(9.573014, -0.402266, 2.365670)),
    'L_UpLeg':((-0.284838, -0.671231, 0.355372, -0.584830),(-11.592800, -3.668512, -1.892024)),
    'R_UpLeg':((-0.595265, 0.235351, 0.741018, 0.202883),(-13.148896, 2.753957, -1.877765)),
    'L_LowLeg':((-0.208370, 0.724054, 0.203874, 0.625111),(10.004286, -5.902038, -21.512903)),
    'R_LowLeg':((0.692465, 0.272623, -0.638688, 0.195566),(-17.756554, 7.571612, -3.879845)),
    'L_Foot':((-0.122545, 0.357691, 0.277336, -0.883247),(-10.045323, -3.793704, 6.952049)),
    'R_Foot':((0.281548, -0.616531, -0.278380, -0.680533),(15.434917, -9.574031, 6.142306))}

imu_bone_angle = {
    'Sternum': [0.0, 0.0, 0.0], 'Pelvis': [1.5868191824388809, -1.2824054739017954, 3.0875089472260506],
    'L_UpArm': [-3.1100395449943656, 0.67854241549909966, -1.7200033330017488], 'R_UpArm': [-0.74461726302100728, 0.81536965131568861, 1.8102727786397874],
    'L_LowArm': [3.0602020521245423, -0.11838304345199442, -1.7000950864510695], 'R_LowArm': [-1.7425147106719878, 0.023220133422421043, 1.8247429730627467],
    'L_UpLeg': [-0.21705700192027516, 0.93838390426051355, 1.812040317036175], 'R_UpLeg': [-2.8747329952976672, -0.55946790807709568, -1.5558110436920511],
    'L_LowLeg': [-0.078980526667041626, 0.66583009941787674, 1.7035318348315576], 'R_LowLeg': [-3.048339575210735, 0.34442702434973066, -1.6195614212641933]}

def draw_image(bone_vector):
    # 將每個數值都除以10
    for key in bone_vector:
        bone_vector[key] = [i / 10 for i in bone_vector[key]]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    pos = fig.add_subplot(111, projection='3d')
    pos.set_xlabel('x',fontdict={'size':15},labelpad=8, color='#f00')  # 設定 x 軸標題
    pos.set_ylabel('y',fontdict={'size':15},labelpad=8, color='#0f0')  # 設定 y 軸標題
    pos.set_zlabel('z',fontdict={'size':15},labelpad=8, color='#00f')  # 設定 z 軸標題

    body = [[bone_vector['Sternum'][0]+bone_vector['Pelvis'][0],bone_vector['Pelvis'][0]],
            [bone_vector['Sternum'][1]+bone_vector['Pelvis'][1],bone_vector['Pelvis'][1]],
            [bone_vector['Sternum'][2]+bone_vector['Pelvis'][2],bone_vector['Pelvis'][2]]]
    # body = [[bone_vector['Pelvis'][0]],
    #         [bone_vector['Pelvis'][1]],
    #         [bone_vector['Pelvis'][2]]]
    left_leg = [[bone_vector['L_UpLeg'][0],bone_vector['L_UpLeg'][0]+bone_vector['L_LowLeg'][0]],
                [bone_vector['L_UpLeg'][1],bone_vector['L_UpLeg'][1]+bone_vector['L_LowLeg'][1]],
                [bone_vector['L_UpLeg'][2],bone_vector['L_UpLeg'][2]+bone_vector['L_LowLeg'][2]]]
    right_leg = [[bone_vector['R_UpLeg'][0],bone_vector['R_UpLeg'][0]+bone_vector['R_LowLeg'][0]],
                [bone_vector['R_UpLeg'][1],bone_vector['R_UpLeg'][1]+bone_vector['R_LowLeg'][1]],
                [bone_vector['R_UpLeg'][2],bone_vector['R_UpLeg'][2]+bone_vector['R_LowLeg'][2]]]
    left_arm = [[bone_vector['L_UpArm'][0],bone_vector['L_UpArm'][0]+bone_vector['L_LowArm'][0]],
                [bone_vector['L_UpArm'][1],bone_vector['L_UpArm'][1]+bone_vector['L_LowArm'][1]],
                [bone_vector['L_UpArm'][2],bone_vector['L_UpArm'][2]+bone_vector['L_LowArm'][2]]]
    right_arm = [[bone_vector['R_UpArm'][0],bone_vector['R_UpArm'][0]+bone_vector['R_LowArm'][0]],
                [bone_vector['R_UpArm'][1],bone_vector['R_UpArm'][1]+bone_vector['R_LowArm'][1]],
                [bone_vector['R_UpArm'][2],bone_vector['R_UpArm'][2]+bone_vector['R_LowArm'][2]]]
    # 繪製 3D 座標點
    pos.scatter(right_leg[0], right_leg[1], right_leg[2], color='red', marker='o', label='Right Leg')
    pos.plot(right_leg[0], right_leg[1], right_leg[2], color='r', label='Right Leg')
    pos.scatter(left_leg[0], left_leg[1], left_leg[2], color='orange', marker='o', label='Left Leg')
    pos.plot(left_leg[0], left_leg[1], left_leg[2], color='orange', label='Left Leg')
    pos.scatter(body[0], body[1], body[2], color='green', marker='x', label='Body')
    pos.plot(body[0], body[1], body[2], color='g', label='Body')
    pos.scatter(right_arm[0], right_arm[1], right_arm[2], color='blue', marker='s', label='Right Limb')
    pos.plot(right_arm[0], right_arm[1], right_arm[2], color='b', label='Right Limb')
    pos.scatter(left_arm[0], left_arm[1], left_arm[2], color='purple', marker='s', label='Left Limb')
    pos.plot(left_arm[0], left_arm[1], left_arm[2], color='purple', label='Left Limb')

    # 顯示圖例
    pos.legend()

    pos.set_xlim([-100, 100])
    pos.set_ylim([-100, 100])
    pos.set_zlim([-100, 100])
    plt.show()

# # 定義你的四元數
# # [w, x, y, z]
# q_Tb = np.array([6.12303177e-17, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00])
# q_Ii = np.array([0.538190, -0.440111, -0.357176, -0.623762])

# # 計算 q_Ii 的共軛
# q_Ii_conj = R.from_quat(np.roll(q_Ii, shift=1)).inv().as_quat()
# q_Ii_conj = np.roll(q_Ii_conj, shift=-1)

# # 計算 q_ib
# q_ib = R.from_quat(np.roll(q_Ii_conj, shift=1)) * R.from_quat(np.roll(q_Tb, shift=1))
# q_ib = q_ib.as_quat()   # [x, y, z, w]
# q_ib = np.roll(q_ib, shift=1)   # [w, x, y, z]

# print(q_ib)

# # calculate T_ib
# # 定義你的frame
# frame = np.array([0, 0, 1])  # 假設你的frame在x軸上

# # 定義你的目標旋轉四元數
# q = [-0.050350, 0.039750, -0.698497, 0.712732]

# # # 注意scipy的四元數格式為(x, y, z, w)，所以我們需要重新排列四元數
# # q = q[-1:] + q[:-1]

# # 創建一個旋轉對象
# r = R.from_quat(q)

# # 使用這個旋轉來旋轉你的frame
# rotated_frame = r.apply(frame)
# euler_angles = r.as_euler('xyz', degrees=True)

# print(euler_angles)
# print(rotated_frame)

# # calculate quaternion(one angle)
# rot_x = 0.0
# rot_y = 0.0
# rot_z = 0.91
# euler_TI = [rot_x, rot_y, rot_z]  # euler angle in degree
# r = R.from_euler('xyz', euler_TI, degrees=True) # 先繞 x 軸旋轉，然後繞 y 軸旋轉，最後繞 z 軸旋轉
# R_TI_qua = r.as_quat()
# R_TI_qua = np.roll(R_TI_qua, shift=1)
# R_TI_mat = r.as_matrix()
# print('=>q_TI_qua:', R_TI_qua)  # [w, x, y, z]
# print('=>R_TI_mat:', R_TI_mat)

# # calculate quaternion(prepare for data)
# rot_vb_x = -90.0
# rot_vb_y = 90.0
# rot_vb_z = 0.0
# euler_vb = [rot_vb_x, rot_vb_y, rot_vb_z]  # euler angle in degree
# r = R.from_euler('xyz', euler_vb, degrees=True) # 先繞 x 軸旋轉，然後繞 y 軸旋轉，最後繞 z 軸旋轉
# R_vb_qua = r.as_quat()
# q_imu_bone_vicon = dict()
# for bone in bones:
#     rot_x = imu_bone_angle[bone][0]
#     rot_y = imu_bone_angle[bone][1]
#     rot_z = imu_bone_angle[bone][2]
#     euler_bi = [rot_x, rot_y, rot_z]  # euler angle in radian
#     r = R.from_euler('xyz', euler_bi, degrees=False) # 先繞 x 軸旋轉，然後繞 y 軸旋轉，最後繞 z 軸旋轉
#     R_bi_qua = r.as_quat()  # [x, y, z, w]
#     q_Tb = R_vb_qua * R_bi_qua
#     # R_bi_qua = np.roll(R_bi_qua, shift=1)
#     q_imu_bone_vicon[bone] = q_Tb
# print('=>q_imu_bone_vicon:', q_imu_bone_vicon)
# with open('imu_bone_vicon.txt', 'w+') as f:
#     f.write(str(q_imu_bone_vicon))

# # calculate R_bi_for_sim
# rot_x = 0.0
# rot_y = 0.0
# rot_z = 0.0
# euler_vb = [rot_x, rot_y, rot_z]  # euler angle in degree
# r = R.from_euler('xyz', euler_vb, degrees=True) # 先繞 x 軸旋轉，然後繞 y 軸旋轉，最後繞 z 軸旋轉
# R_vb_qua = r.as_quat()
# R_vb_qua = np.roll(R_vb_qua, shift=1)   # [w, x, y, z]
# rot_angle_radian = dict()
# for bone in bones:
#     q_bi_data = calib_imu_bone[bone]
#     q_bi_data = Quaternion(q_bi_data)
#     q_vb = Quaternion(R_vb_qua)
#     q_bv = q_vb.conjugate
#     # print('=>q_bv:', q_bv)
#     q_bi_for_sim = q_bv * q_bi_data
#     euler_bi_for_sim = R.from_quat([q_bi_for_sim.x, q_bi_for_sim.y, q_bi_for_sim.z, q_bi_for_sim.w])
#     euler_bi_for_sim = euler_bi_for_sim.as_euler('xyz', degrees=False)
#     # rot_angle_radian[bone] = euler_bi_for_sim
#     print('=>euler_bi_for_sim_', bone, ':', euler_bi_for_sim)

# q_pel_data = calib_imu_bone['Pelvis']
# q_pel_data = Quaternion(q_pel_data)
# for bone in bones:
#     if bone == 'Pelvis':
#         q_data = q_pel_data
#     else:
#         q_data = calib_imu_bone[bone]
#         q_data = Quaternion(q_data)
#         q_data = q_data * q_pel_data.conjugate
#     euler_data = R.from_quat([q_data.x, q_data.y, q_data.z, q_data.w])
#     euler_data = euler_data.as_euler('xyz', degrees=False)
#     print('=>euler_', bone, '_for_sim:', euler_data)

# # calculate euler angle
# euler_angles = dict()
# for bone in bones:
#     euler = dict()

    # # 四元數轉成旋轉矩陣再轉成尤拉角
    # q_TI = calib_imu_ref[bone]
    # # print('q_TI_euler_angle:', quaternion_to_euler_angles(q_TI))
    # q_TI = Quaternion(q_TI)
    # q_TI_matrix = q_TI.rotation_matrix
    # TI_euler_angles = rotation_matrix_to_euler_angles(q_TI_matrix)
    # TI_euler_angles_degree = np.degrees(TI_euler_angles)
    # euler['TI'] = TI_euler_angles_degree

    # # 四元數轉成尤拉角
    # q_TI = calib_imu_ref[bone]
    # q_TI = Quaternion(q_TI)
    # r_TI = R.from_quat([q_TI.x, q_TI.y, q_TI.z, q_TI.w])
    # euler_TI = r_TI.as_euler('xyz', degrees=True)  # 先繞 x 軸旋轉，然後繞 y 軸旋轉，最後繞 z 軸旋轉
    # euler['TI'] = euler_TI

    # q_bi = calib_imu_bone[bone]
    # q_bi = Quaternion(q_bi)
    # r_bi = R.from_quat([q_bi.x, q_bi.y, q_bi.z, q_bi.w])
    # euler_bi = r_bi.as_euler('xyz', degrees=True)  # 先繞 x 軸旋轉，然後繞 y 軸旋轉，最後繞 z 軸旋轉
    # euler['bi'] = euler_bi

    # q_ib = q_bi.conjugate
    # r_ib = R.from_quat([q_ib.x, q_ib.y, q_ib.z, q_ib.w])
    # euler_ib_xyz = r_ib.as_euler('xyz', degrees=True)  # 先繞 x 軸旋轉，然後繞 y 軸旋轉，最後繞 z 軸旋轉
    # euler_ib_zyx = r_ib.as_euler('zyx', degrees=True)  # 先繞 z 軸旋轉，然後繞 y 軸旋轉，最後繞 x 軸旋轉
    # euler['ib_xyz'] = euler_ib_xyz
    # euler['ib_zyx'] = euler_ib_zyx

    # ori = imu_data[bone][0]
    # q_Ii = Quaternion(ori)
    # r_Ii = R.from_quat([q_Ii.x, q_Ii.y, q_Ii.z, q_Ii.w])
    # euler_Ii = r_Ii.as_euler('xyz', degrees=True)  # 先繞 x 軸旋轉，然後繞 y 軸旋轉，最後繞 z 軸旋轉
    # euler['Ii'] = euler_Ii

    # q_Tb = q_Ii * q_ib
    # r_Tb = R.from_quat([q_Tb.x, q_Tb.y, q_Tb.z, q_Tb.w])
    # euler_Tb = r_Tb.as_euler('xyz', degrees=True)  # 先繞 x 軸旋轉，然後繞 y 軸旋轉，最後繞 z 軸旋轉
    # euler['Tb'] = euler_Tb

#     euler_angles[bone] = euler
# print('=>euler_angles:', euler_angles)

# calculate bone vector
bone_vector_with_qTi = dict()
for bone in bones:
    bone_vec = np.array(bone_info[bone][2]) * 25.4  # length of bone * 25.4
    q_TI = calib_imu_ref[bone]
    q_bi = calib_imu_bone[bone]
    q_TI = Quaternion(q_TI)
    q_bi = Quaternion(q_bi)
    q_ib = q_bi.conjugate
    ori = imu_data[bone][0]
    q_Ii = Quaternion(ori)
    q_Tb = q_TI * q_Ii * q_ib
    rotated_bone_vec = q_Tb.rotate(bone_vec)
    bone_vector_with_qTi[bone] = rotated_bone_vec
print('=>bone_vector_with_qTi:', bone_vector_with_qTi)
draw_image(bone_vector_with_qTi)

# rot_vb_x = 0.0
# rot_vb_y = 0.0
# rot_vb_z = 0.0
# euler_vb = [rot_vb_x, rot_vb_y, rot_vb_z]  # euler angle in degree
# r = R.from_euler('xyz', euler_vb, degrees=True) # 先繞 x 軸旋轉，然後繞 y 軸旋轉，最後繞 z 軸旋轉
# R_vb_qua = Quaternion(r.as_quat())
# bone_vector_no_qTi = dict()
# for bone in bones:
#     bone_vec = np.array(bone_info[bone][2]) * 25.4  # length of bone * 25.4
#     q_bi = calib_imu_bone[bone]
#     q_bi = Quaternion(q_bi)
#     q_ib = q_bi.conjugate
#     ori = imu_data[bone][0]
#     q_Ii = Quaternion(ori)
#     if rot_vb_x == 0.0 and rot_vb_y == 0.0 and rot_vb_z == 0.0:
#         q_Tb = q_Ii * q_ib
#     else:
#         q_Tb = R_vb_qua * q_Ii * q_ib
#     rotated_bone_vec = q_Tb.rotate(bone_vec)
#     bone_vector_no_qTi[bone] = rotated_bone_vec
# print('=>bone_vector_no_qTi:', bone_vector_no_qTi)
# draw_image(bone_vector_no_qTi)

# bone_vector_only_qib = dict()
# for bone in bones:
#     bone_vec = np.array(bone_info[bone][2]) * 25.4  # length of bone * 25.4
#     q_bi = calib_imu_bone[bone]
#     q_bi = Quaternion(q_bi)
#     q_ib = q_bi.conjugate
#     q_Tb = q_ib
#     rotated_bone_vec = q_Tb.rotate(bone_vec)
#     bone_vector_only_qib[bone] = rotated_bone_vec
# print('=>bone_vector_only_qib:', bone_vector_only_qib)

# bone_vector_only_qIi = dict()
# for bone in bones:
#     bone_vec = np.array(bone_info[bone][2]) * 25.4  # length of bone * 25.4
#     ori = imu_data[bone][0]
#     q_Ii = Quaternion(ori)
#     q_Tb = q_Ii
#     rotated_bone_vec = q_Tb.rotate(bone_vec)
#     bone_vector_only_qIi[bone] = rotated_bone_vec
# print('=>bone_vector_only_qIi:', bone_vector_only_qIi)
# draw_image(bone_vector_only_qIi)