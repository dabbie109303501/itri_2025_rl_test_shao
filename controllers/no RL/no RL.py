# import numpy as np
# from controller import Robot, Supervisor,Connector
# from ikpy.chain import Chain
# import pandas as pd
# import numpy as np
# from gymnasium import Env
# from gymnasium import spaces
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from scipy.spatial.transform import Rotation as R #如果需要將旋轉矩陣轉換為 四元數 (quaternion) 或 歐拉角 (Euler angles)，可以使用
# from collections import defaultdict
# import pandas as pd
# import random

# # 全域或類別屬性都可以
# supervisor = Supervisor()

# robot_chain = Chain.from_urdf_file("LRMate-200iD.urdf",base_elements=['Base'])


# #利用順項運動學計算出末端軸位置
# def get_endpoint_position(angles):#input為機器手臂各軸馬達的位置(角度，以弧度表示)
#     endpoint_position=robot_chain.forward_kinematics(angles)
#     return endpoint_position #output為手臂末端軸位置(以轉至矩陣表示)

# def get_IK_angle(target_position, target_orientation,initial_position, orientation_axis="all"):
#     """
#     计算逆向运动学角度
#     :param target_position: 目标末端位置 [x, y, z]
#     :param target_orientation: 目标方向 (3x3 旋转矩阵)
#     :param orientation_axis: 指定对齐轴 ("x", "y", "z")，或者 "all" 进行完整姿态匹配
#     :return: 6 轴角度 (弧度制)
#     :initial_position: 手臂各軸的初始角度 [a1,a2,a3,a4,a5,a6]
#     """
#     # Initial_Position = np.zeros(10) ###
#     # Initial_Position[2:8]=initial_position ###

#     # 1. 先複製一份 mask，並排除 Base link（index 0）
#     mask = robot_chain.active_links_mask.copy()
#     mask[0] = False

#     # 2. 建一個與 mask 同長度的零向量
#     init_full = np.zeros(mask.shape[0])
#     # 3. 只在 mask==True 的位置，填入你的 6 軸 initial_position
#     init_full[mask] = initial_position


#     # 计算逆运动学
#     ik_angles = robot_chain.inverse_kinematics(
#         target_position,
#         target_orientation=target_orientation ,
#         orientation_mode=orientation_axis,
#         initial_position=init_full
#     )
#     # return ik_angles[2:8] # 取 6 轴角度 (去掉基座)
#     # 4. **穩健地**用 mask 把 active joints 抽出來
#     target_angles = ik_angles[mask]

#     # 5. 確認數量正確（可選）
#     assert target_angles.shape[0] == len(initial_position), \
#         f"IK 回傳 {target_angles.shape[0]} 軸，但環境有 {len(initial_position)} 軸"

#     return target_angles

# def axis_angle_to_quaternion(axis, angle):
#     """
#     將軸角表示法 (axis, angle) 轉換為四元數。
#     :param axis: 旋轉軸 (3D 向量)
#     :param angle: 旋轉角度 (弧度)
#     :return: 四元數 [x, y, z, w]
#     """
#     axis = np.array(axis) / np.linalg.norm(axis)  # 確保軸是單位向量
#     quaternion = R.from_rotvec(axis * angle).as_quat()
#     return quaternion

# def quaternion_to_matrix(quaternion):
#     """
#     將四元數轉換為旋轉矩陣。
#     :param quaternion: 四元數 [x, y, z, w]
#     :return: 3x3 旋轉矩陣
#     """
#     r = R.from_quat(quaternion)
#     return r.as_matrix()  

# def matrix_to_quaternion(R_mat):
#     """將旋轉矩陣轉換為四元數"""
#     return R.from_matrix(R_mat).as_quat()

# def compute_target_end_effector_pose(p_w, q_w, p_e, q_e, p_w_new, q_w_new):
#         """
#         根據工件的初始和目標位姿計算新的末端執行器位姿。
#         輸入工件與末端執行器的初始位置以確定兩坐標系之間的相對位置，接著就可以根據工件的目標位置推算出相對應末端執行器的位置
#         :param p_w: 初始工件位置 (x, y, z)
#         :param q_w: 初始工件四元數 (x, y, z, w)
#         :param p_e: 初始末端執行器位置 (x, y, z)
#         :param q_e: 初始末端執行器四元數 (x, y, z, w)
#         :param p_w_new: 新的工件位置 (x, y, z)
#         :param q_w_new: 新的工件四元數 (x, y, z, w)
#         :return: (p_e_new, q_e_new) 新的末端執行器位置和四元數
#         """
#         # 將四元數轉換為旋轉矩陣
#         R_w = quaternion_to_matrix(q_w)
#         R_e = quaternion_to_matrix(q_e)
#         R_w_new = quaternion_to_matrix(q_w_new)
        
#         # 轉換為齊次變換矩陣
#         T_w = np.eye(4)
#         T_w[:3, :3] = R_w
#         T_w[:3, 3] = p_w
        
#         T_e = np.eye(4)
#         T_e[:3, :3] = R_e
#         T_e[:3, 3] = p_e
        
#         # 計算工件相對於末端執行器的變換矩陣
#         T_we = np.linalg.inv(T_w) @ T_e
        
#         # 計算新的工件變換矩陣
#         T_w_new = np.eye(4)
#         T_w_new[:3, :3] = R_w_new
#         T_w_new[:3, 3] = p_w_new
        
#         # 計算新的末端執行器變換矩陣
#         T_e_new = T_w_new @ T_we
        
#         # 提取新的位置與旋轉矩陣
#         p_e_new = T_e_new[:3, 3]
#         R_e_new = T_e_new[:3, :3]
        
#         # 轉換為四元數
#         q_e_new = matrix_to_quaternion(R_e_new)
        
#         return p_e_new, q_e_new

# def directly_go_to_target(quaternion,p_w_new,robot_initial_pos):
#     """
#     輸入工件的目標位置以及目標角度(以軸角表示法表示),以及手臂的初始姿態
#     quaternion:工件的目標角度(以四位元數表示)
#     p_w_new:工件目標位置
#     axis:旋轉軸
#     angle:旋轉角
#     robot_initial_pos:手臂各軸的初始角度 [a1,a2,a3,a4,a5,a6]
#     """
#     #輸入工件目標位置的旋轉(軸角表示法)，將其轉換為四位元數
#     # axis = [0.654636,0.377894,-0.654712]
#     # angle = 2.41868
#     #輸入工件目標位置的平移與旋轉()
#     # p_w_new = np.array([0.059, 0.646998, 0.46])
#     q_w_new = np.array(quaternion)

#     #輸入工件與末端執行器初始位置(四位元數表示旋轉)，根據工件目標位置計算末端執行器的目標位置
#     # p_w = np.array([-0.488223, 0.000823146, 0.000823146])
#     p_w = np.array([-0.488223, 0.000823146, 0.68174])
#     q_w = axis_angle_to_quaternion([-0.576282, 0.578236, 0.57753], 2.09629)
#     # q_w = np.array([0.499171, -0.499340, 0.501038, 0.500449]) #四位元數
#     # p_e = np.array([-0.425723, 0.000796888, 0.682031])
#     p_e = np.array([-0.425741, 0.000827394, 0.681564])
#     q_e = axis_angle_to_quaternion([-0.00119838, -0.707383, -0.706829], 3.13982)
#     # q_e= np.array([  0.00090633,  -0.00120166,  -0.70737343,  -0.70683843  ]) #四位元數
    
#     p_e_new, q_e_new = compute_target_end_effector_pose(p_w, q_w, p_e, q_e, p_w_new, q_w_new)#末端執行器的目標位置

#     R_e_new = quaternion_to_matrix(q_e_new)#將四位元數轉換成旋轉矩陣

#     ikAnglesD=get_IK_angle(p_e_new,R_e_new,robot_initial_pos)#利用逆向運動學求出機器手臂各軸角度

#     return ikAnglesD

# def get_motor_angles(self):
#     #各馬達上加入sensor測量轉角位置
#     # sensors = []
#     # for motor in self.motors:
#     #     sensor = motor.getPositionSensor()
#     #     # sensor.enable(timestep)
#     #     sensors.append(sensor)

#     joint_angles = [sensor.getValue() for sensor in self.sensors]#讀取各軸位置

#     return (np.array(joint_angles))



# motors = []
# motors.append(supervisor.getDevice('J1'))
# motors.append(supervisor.getDevice('J2'))
# motors.append(supervisor.getDevice('J3'))
# motors.append(supervisor.getDevice('J4'))
# motors.append(supervisor.getDevice('J5'))
# motors.append(supervisor.getDevice('J6'))

# # current_angles = self.get_motor_angles()   # shape (6,)

# '''
# 讀取路徑採樣點資訊後，採樣點位置為相對於工件坐標系之坐標系，將採樣點的坐標系(在工件坐標系底下)的旋轉與平移，以及砂帶接觸點(世界坐標系底下)的旋轉與平移作為輸入，
# 輸出工件坐標系在世界坐標系底下的平移(t_A_prime)與旋轉(R_A_prime)
# calculate_A_prime:輸入工件路徑採樣點研磨時的座標平移與歐拉角，輸出工件座標研磨時的平移與歐拉角
# '''
# #砂帶接觸點(世界坐標系底下)的旋轉與平移

# #砂帶上接觸點的座標
# t_contactpoint_frame = np.array([0.600058, -1.25244e-05, 0.638517])
# #砂帶上接觸點的座標的旋轉
# axis = np.array([-1.85197e-06, -1.0, -8.81442e-06]) 
# theta = 2.0944 
# R_contactpoint_frame = quaternion_to_matrix( axis_angle_to_quaternion(axis, theta) ) #將軸角表示法轉為旋轉矩陣

# joints_angle = get_IK_angle(t_contactpoint_frame, R_contactpoint_frame, initial_position=[0,0,0,0,0,0]) #利用逆向運動學求出機器手臂各軸角度 
# for motor, ang in zip(motors, joints_angle):
#     motor.setPosition(float(ang))

#----------------------------------------
# from controller import Supervisor
# from ikpy.chain import Chain
# import math
# import random
# from ikpy.utils import geometry
# import numpy as np

# # 初始化
# sup = Supervisor()
# timestep = int(sup.getBasicTimeStep())

# # 取得 Node
# effector = sup.getFromDef("J6")
# target   = sup.getFromDef("BELT_TARGET")


# # 載入 IK Chain（預先準備好 URDF）
# chain = Chain.from_urdf_file("LRMate-200iD.urdf",base_elements=['Base'])


# # 取得所有關節 motor 裝置
# motor_names = ["J1", "J2", "J3", "J4", "J5", "J6"]
# motors = [ sup.getDevice(name) for name in motor_names ]
# for m in motors:
#     m.setPosition(0.0)    # 初始化位置
#     m.setVelocity(1.0)    # 設定最大速度

# def axis_angle_to_matrix(axis, angle):
#     """
#     把 axis‐angle 轉成 3×3 旋轉矩陣。

#     參數:
#     - axis: 長度為 3 的 list 或 ndarray，表示旋轉軸向量 (ax, ay, az)。
#     - angle: 浮點數 (rad)，表示要繞該軸旋轉的角度。

#     回傳:
#     - R: numpy.ndarray，shape=(3,3)，對應的旋轉矩陣。
#     """
#     u = np.array(axis, dtype=float)
#     norm = np.linalg.norm(u)
#     if norm == 0:
#         # 若輸入軸向量為零，就回傳單位矩陣
#         return np.eye(3)
#     u = u / norm  # 單位化
#     ux, uy, uz = u[0], u[1], u[2]

#     c = np.cos(angle)
#     s = np.sin(angle)
#     one_c = 1 - c

#     # Rodrigues' 公式
#     R = np.array([
#         [c + ux*ux*one_c,       ux*uy*one_c - uz*s,  ux*uz*one_c + uy*s],
#         [uy*ux*one_c + uz*s,    c + uy*uy*one_c,     uy*uz*one_c - ux*s],
#         [uz*ux*one_c - uy*s,    uz*uy*one_c + ux*s,  c + uz*uz*one_c   ]
#     ])
#     return R

# while sup.step(timestep) != -1:


#     # 1. 讀取目標座標
#     # tgt = target.getField("translation").getSFVec3f()
#     tgt = [0.614, -0.002, 0.613]  # 跟砂帶接觸點x差了一個負號
#     tgr = target.getField("rotation").getSFRotation()
#     print("Target Position:", tgt)
#     print("Target Rotation:", tgr)

#     # 2. IK 計算
#     angles = chain.inverse_kinematics(tgt, tgr)

#     # 3. 將計算得到的角度輸入各關節
#     for i, m in enumerate(motors):
#         m.setPosition(angles[i+1])

#     eef = sup.getFromDef("J6")
#     final = eef.getField("translation").getSFVec3f()
#     # print("End Effector Position:", final)
#     # print("End Effector Position:", eef.getPosition())

#-----------------------------------------------
# ---------------------------------------------------------------------------------
# controller： sample_alignment_controller.py
# 功能：依序把「工件座標系下的多組採樣點」對齊到同一個世界座標目標
# ---------------------------------------------------------------------------------

from controller import Supervisor
from ikpy.chain import Chain
import numpy as np
import pandas as pd
import math

# ---------- 1. 初始化 Supervisor 與 IKPY ----------
sup      = Supervisor()
timestep = int(sup.getBasicTimeStep())

# 1.1 抓取「末端 J6」和「目標 BELT_TARGET」在世界裡的節點
j6_node   = sup.getFromDef("J6")
target    = sup.getFromDef("BELT_TARGET")   # 目標點 (含 rotation)
# target = [0.614, -0.002, 0.613]  # 砂帶接觸點 x 差了一個負號

if (j6_node is None) or (target is None):
    print("Error: 找不到 J6 或 BELT_TARGET 節點，請檢查 DEF 名稱！")
    exit(1)

# 1.2 載入 IKPY Chain (確保 base_elements 與 URDF 裡的 root link 名稱一致)
chain = Chain.from_urdf_file("LRMate-200iD.urdf", base_elements=['Base'])

# 1.3 抓 Webots 裡 J1~J6 的 motor（一定要跟 world 裡 HingeJoint 下的 name 一致）
motor_names = ["J1", "J2", "J3", "J4", "J5", "J6"]
motors = []
for name in motor_names:
    m = sup.getDevice(name)
    if m is None:
        print(f"Error: 無法取得馬達 '{name}'，請確認 world 檔內的 RotationalMotor name。")
        exit(1)
    motors.append(m)
    m.setPosition(0.0)
    m.setVelocity(1.0)  # 初始先給個小速度，後面可再調

# ---------- 2. 計算「工件→末端」固定變換  {}^E T_C ----------
#    假設工件在 world 裡已經用 DEF WORKPIECE 標示
workpiece = sup.getFromDef("FLAT")
if workpiece is None:
    print("Error: 找不到 DEF WORKPIECE 節點！")
    exit(1)

# 2.1 讀取工件（C）與 J6（E）在世界座標下的 translation / rotation (axis‐angle)
wp_t = workpiece.getPosition()   # e.g. [x_w, y_w, z_w]
wp_r = workpiece.getOrientation()   # e.g. [ax, ay, az, θ]
# print("Workpiece Position:", wp_t)

j6_t = j6_node.getPosition() 
j6_r = j6_node.getOrientation()
# print("End Effector Position:", j6_t)

# 2.2 自製一個「axis‐angle → 3×3 旋轉矩陣」的小函式
def axis_angle_to_matrix(axis, angle):
    u = np.array(axis, dtype=float)
    norm = np.linalg.norm(u)
    if norm == 0:
        return np.eye(3)
    u = u / norm
    ux, uy, uz = u[0], u[1], u[2]
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1 - c
    R = np.array([
        [c + ux*ux*one_c,       ux*uy*one_c - uz*s,  ux*uz*one_c + uy*s],
        [uy*ux*one_c + uz*s,    c + uy*uy*one_c,     uy*uz*one_c - ux*s],
        [uz*ux*one_c - uy*s,    uz*uy*one_c + ux*s,  c + uz*uz*one_c   ]
    ])
    return R

def rotation_matrix_to_quaternion(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        S  = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S  = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S  = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S  = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    return [qx, qy, qz, qw]

# 2.3 組出 {}^W T_C 與 {}^W T_E
R_wp = axis_angle_to_matrix(wp_r[0:3], wp_r[3])
T_wp = np.eye(4)
T_wp[0:3, 0:3] = R_wp
T_wp[0:3, 3]   = np.array(wp_t)

R_j6 = axis_angle_to_matrix(j6_r[0:3], j6_r[3])
T_j6 = np.eye(4)
T_j6[0:3, 0:3] = R_j6
T_j6[0:3, 3]   = np.array(j6_t)

# 2.4 求出末端到工件的固定剛體變換 {}^E T_C = ({}^W T_E)^(-1) * ({}^W T_C)
T_e_inv = np.linalg.inv(T_j6)
T_E_C   = T_e_inv.dot(T_wp)
# 把 T_E_C 拆成旋轉矩陣 R_EC（3×3）與平移向量 t_EC（3×1）
R_EC = T_E_C[0:3, 0:3]
t_EC = T_E_C[0:3, 3]

print("J6 world translation:", j6_node.getPosition())
print("J6 world orientation matrix:\n", np.array(j6_node.getOrientation()).reshape(3,3))
print("Workpiece world translation:", workpiece.getPosition())
print("Workpiece world orientation matrix:\n", np.array(workpiece.getOrientation()).reshape(3,3))
print("Computed T_E_C:\n", T_E_C)

# ---------- 3. 讀取 flat.csv 裡所有採樣點（工件座標系）----------
df = pd.read_csv('flat.csv', header=None)
sampling_points = []
for _, row in df.iterrows():
    x_c = float(row[2]) / 1000.0   # 單位轉成 m
    y_c = float(row[3]) / 1000.0
    z_c = float(row[4]) / 1000.0
    sampling_points.append([x_c, y_c, z_c])
# sampling_points = [ [x1,y1,z1], [x2,y2,z2], … ]

# ---------- 4. 主迴圈：依序把每個採樣點對齊到 BELT_TARGET ----------
for p_C in sampling_points:
    # 假設 BELT_TARGET 是一個在場景裡「固定不動」的節點，
    # 而且我們要讓「工件上的 p_C」這個點飄到它的世界座標 target_pos
    # 同時讓該點的法向與 BELT_TARGET 的朝向一致。

    # 4.1 讀 BELT_TARGET 在世界裡的 translation/rotation
    tgt_world_pos = target.getField("translation").getSFVec3f()   # [x_w, y_w, z_w]
    tgt_world_rot = target.getField("rotation").getSFRotation()   # [ax, ay, az, θ]

    # 4.2 把 BELT_TARGET 這個「目標節點」改成 4×4 同質矩陣 {}^W T_T
    #     用剛剛的 axis_angle_to_matrix 轉成 3×3 R_WT
    R_WT = axis_angle_to_matrix(tgt_world_rot[0:3], tgt_world_rot[3])
    T_W_T = np.eye(4)
    T_W_T[0:3, 0:3] = R_WT
    T_W_T[0:3, 3]   = np.array(tgt_world_pos)

    # 4.3 構造「工件座標系下某採樣點」到工件原點的逆變換 {}^C T_p ^(-1)
    #     因為點 p_C 在工件座標下的同質矩陣是 [I, p_C; 0,1]，
    #     取逆就是 [I, -p_C; 0,1]
    T_C_p_inv = np.eye(4)
    T_C_p_inv[0:3, 3] = -np.array(p_C)

    # 4.4 現在就有：
    #      {}^W T_E_desired 
    #    = {}^W T_T          × {}^C T_p ^(-1)    × ({}^E T_C)^(-1)
    #    = T_W_T             @ T_C_p_inv         @ np.linalg.inv(T_E_C)
    T_C_E = np.linalg.inv(T_E_C)   # ({}^E T_C)^(-1)
    T_W_E_des = T_W_T.dot(T_C_p_inv).dot(T_C_E)

    # 4.5 把 T_W_E_des 拆成目標「末端位置」與「末端朝向 (axis‐angle)」
    #     – 位置就在 T_W_E_des[0:3,3]；旋轉矩陣就在 T_W_E_des[0:3,0:3]
    R_W_E_des = T_W_E_des[0:3, 0:3]
    pos_W_E_des = T_W_E_des[0:3, 3]

    #    把 3×3 旋轉矩陣 R_W_E_des 再轉回 axis‐angle
    #    這裡示範簡單的 Euler→axis-angle or 四元數→axis-angle 轉換，
    #    也可以直接用 IKPY 的 helper function（若版本支援）。
    def rotation_matrix_to_axis_angle(R):
        # 「三角公式」從 R 反推 axis‐angle
        # 參考 https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_rotation_matrix_to_axis%E2%80%93angle
        angle = math.acos((np.trace(R) - 1) / 2)
        if abs(angle) < 1e-6:
            return [1,0,0, 0.0]  # 若 angle 約 0，直接回傳任何軸 + 0 弧度
        # 否則軸 = (1/(2 sin θ)) (R32 - R23, R13 - R31, R21 - R12)
        ux = (R[2,1] - R[1,2]) / (2 * math.sin(angle))
        uy = (R[0,2] - R[2,0]) / (2 * math.sin(angle))
        uz = (R[1,0] - R[0,1]) / (2 * math.sin(angle))
        return [ux, uy, uz, angle]

    des_axis_angle = rotation_matrix_to_axis_angle(R_W_E_des)

    # 4.6 把「末端目標位姿」(pos_W_E_des + des_axis_angle) 送進 IKPY → 得到關節角度
    #     IKPY 預設 inverse_kinematics 需要一個 4×4 同質矩陣，
    #     我們先把 pos 跟 axis‐angle 再轉回 4×4
    T_des_frame = np.eye(4)
    T_des_frame[0:3, 0:3] = R_W_E_des
    T_des_frame[0:3, 3]   = pos_W_E_des

    R_W_E_des   = T_des_frame[0:3, 0:3]
    pos_W_E_des = T_des_frame[0:3, 3]
    quat = rotation_matrix_to_quaternion(R_W_E_des)

    angles = chain.inverse_kinematics(
        target_position    = pos_W_E_des,
        target_orientation = quat
    )

    # 3. 把角度下到各 motor
    for i, m in enumerate(motors):
        m.setPosition(angles[i+1])

    # 4.7 把 angles[i+1] 下發給每顆馬達 (i=0 → J1, … i=5 → J6)
    for i, m in enumerate(motors):
        m.setPosition(angles[i+1])

    # 4.8 最後短暫延遲一下（或偵測末端平移誤差 < 某容限後，才繼續下個採樣點）
    #     這裡示範最簡單的 time.sleep，也可以改成「直到末端到位」再繼續。
    sup.step(timestep)
    # 如果要等到末端實際到位才下一個採樣點，可加上：
    # while True:
    #     sup.step(timestep)
    #     curr_pos = j6_node.getField("translation").getSFVec3f()
    #     dist = math.dist(curr_pos, pos_W_E_des)
    #     if dist < 0.005:  # 5 mm 以內就可認為到位
    #         break

# 迴圈結束：所有採樣點都已經依序對齊到 BELT_TARGET
print("所有採樣點已經對齊完成！")

