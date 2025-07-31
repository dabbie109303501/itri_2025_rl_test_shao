from controller import Supervisor,Connector
from ikpy.chain import Chain
import pandas as pd
import numpy as np
import time
import random
import cv2
import math
import random
import csv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ikpy.chain")


supervisor = Supervisor()
# # get the time step of the current world.
robot_chain = Chain.from_urdf_file("LRMate-200iD_try_fix_arm_itristl.urdf",base_elements=['Base'])
# # Ensure Base link (index 0) is not included in the active links mask
# # active_links_mask = [True] * len(robot_chain.links)  # Set all links as active
# # active_links_mask[0] = False  # Exclude the Base link
# print(robot_chain.links)
timestep = int(supervisor.getBasicTimeStep())

#利用順項運動學計算出末端軸位置
def get_endpoint_position(angles):
    endpoint_position=robot_chain.forward_kinematics(angles)
    return endpoint_position

#利用逆向運動學，根據末端軸位置以及角度推算手臂個軸角度
def get_IK_angle(target_position, target_orientation, orientation_axis="all",starting_nodes_angles=[0,0,0,0,0,0,0]): 
    # 初始化機器人鏈條
    
    # for i, link in enumerate(robot_chain.links):
    #     print(f"Joint {i}: {link.name}")
    # 計算逆向運動學
    # ikAnglesD = robot_chain.inverse_kinematics(target_position,target_orientation=[0, 1, 0])#不限制角度
    ikAnglesD= robot_chain.inverse_kinematics(
    target_position,
    target_orientation=target_orientation,
    orientation_mode=orientation_axis,
    initial_position=starting_nodes_angles,
    max_iter=200,
    )#限制角度以及末端軸位置
    
    return ikAnglesD

#將旋轉矩陣轉換為歐拉角
def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (in degrees) with ZYX order.
    
    :param R: np.ndarray, a 3x3 rotation matrix.
    :return: tuple of Euler angles (rx, ry, rz) in degrees.
    """
    # Check for valid rotation matrix
    if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("Invalid rotation matrix")
    
    # Extract angles
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0
    
    # Convert radians to degrees
    rx = math.degrees(rx)
    ry = math.degrees(ry)
    rz = math.degrees(rz)
    
    return rx, ry, rz

#輸入工件路徑採樣點研磨時的座標平移與歐拉角，輸出工件座標研磨時的平移與歐拉角
def calculate_A_prime(R_BA, t_BA, R_B_prime, t_B_prime):
    """
    根据 B' 座标系在世界座标系下的表示，计算 A' 在世界座标系下的表示。

    :param R_BA: np.ndarray, B 座标系在 A 座标系下的旋转矩阵 (3x3).
    :param t_BA: np.ndarray, B 座标系在 A 座标系下的平移向量 (3x1).
    :param R_B_prime: np.ndarray, B' 座标系在世界座标系下的旋转矩阵 (3x3).
    :param t_B_prime: np.ndarray, B' 座标系在世界座标系下的平移向量 (3x1).
    :return: (R_A_prime, t_A_prime), A' 座标系在世界座标系下的旋转矩阵和平移向量.
    """
    # 计算 A' 的旋转矩阵
    R_A_prime = R_B_prime @ np.linalg.inv(R_BA)
    
    # 计算 A' 的平移向量
    t_A_prime = t_B_prime - R_A_prime @ t_BA
    R_A_prime=rotation_matrix_to_euler_angles(R_A_prime)
    return R_A_prime, t_A_prime

#將歐拉角轉換為旋轉矩陣
def Rotation_matrix(rx,ry,rz):
    # Convert degrees to radians
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)
    
    # Calculate the rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(rx), -math.sin(rx)],
                    [0, math.sin(rx), math.cos(rx)]])
    
    R_y = np.array([[math.cos(ry), 0, math.sin(ry)],
                    [0, 1, 0],
                    [-math.sin(ry), 0, math.cos(ry)]])
    
    R_z = np.array([[math.cos(rz), -math.sin(rz), 0],
                    [math.sin(rz), math.cos(rz), 0],
                    [0, 0, 1]])
    
    # Combine the rotation matrices
    R = np.dot(np.dot(R_z, R_y), R_x)
    return R

#欧拉角轉换为轴-角表示(webots內的rotation以轴-角表示)
def euler_to_axis_angle(rx, ry, rz):
    """
    Converts Euler angles (in degrees) to axis-angle representation.
    
    Args:
    rx: Rotation around x-axis in degrees.
    ry: Rotation around y-axis in degrees.
    rz: Rotation around z-axis in degrees.
    
    Returns:
    A tuple of four values representing the axis-angle (axis_x, axis_y, axis_z, angle).
    """
    # Convert degrees to radians
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)
    
    # Calculate the rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(rx), -math.sin(rx)],
                    [0, math.sin(rx), math.cos(rx)]])
    
    R_y = np.array([[math.cos(ry), 0, math.sin(ry)],
                    [0, 1, 0],
                    [-math.sin(ry), 0, math.cos(ry)]])
    
    R_z = np.array([[math.cos(rz), -math.sin(rz), 0],
                    [math.sin(rz), math.cos(rz), 0],
                    [0, 0, 1]])
    
    # Combine the rotation matrices
    R = np.dot(np.dot(R_z, R_y), R_x)
    
    # Calculate the axis-angle representation
    angle = math.acos((np.trace(R) - 1) / 2)
    sin_angle = math.sin(angle)
    
    if sin_angle > 1e-6:  # Avoid division by zero
        axis_x = (R[2, 1] - R[1, 2]) / (2 * sin_angle)
        axis_y = (R[0, 2] - R[2, 0]) / (2 * sin_angle)
        axis_z = (R[1, 0] - R[0, 1]) / (2 * sin_angle)
    else:
        # If the angle is very small, the axis is not well-defined, return the default axis
        axis_x = 1
        axis_y = 0
        axis_z = 0

    return axis_x, axis_y, axis_z, angle
#------------------------------------------------------工件姿態反推手臂末端軸姿態

def get_transformation_matrix(rotation, translation):
    """
    根據旋轉矩陣和平移向量生成4x4的齊次轉換矩陣
    """
    T = np.eye(4)
    T[:3, :3] = rotation  # 設置旋轉部分
    T[:3, 3] = translation  # 設置平移部分
    return T

def invert_transformation_matrix(T):
    """
    反轉4x4齊次轉換矩陣
    """
    R_inv = T[:3, :3].T  # 旋轉矩陣的轉置
    t_inv = -R_inv @ T[:3, 3]  # 平移向量的反轉
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def calculate_B_prime(A_rot, A_trans, B_rot, B_trans, A_prime_rot, A_prime_trans):#從
    """
    計算B'的旋轉與平移矩陣
    給予一個A坐標系與B坐標系相對世界座標的轉移矩陣,接著給出A座標系移動後的新座標系A',假設B'相對A'位置與B相對A相同,輸入A,B,A'的旋轉以及平移矩陣,求出B'的平移與旋轉矩陣

    """
    # Step 1: 計算齊次轉換矩陣
    T_A_W = get_transformation_matrix(A_rot, A_trans)
    T_B_W = get_transformation_matrix(B_rot, B_trans)
    T_A_prime_W = get_transformation_matrix(A_prime_rot, A_prime_trans)

    # Step 2: 計算T_B_A
    T_A_W_inv = invert_transformation_matrix(T_A_W)
    T_B_A = T_A_W_inv @ T_B_W

    # Step 3: 計算T_B_prime_W
    T_B_prime_W = T_A_prime_W @ T_B_A

    # Step 4: 提取B'的旋轉和平移矩陣
    B_prime_rot = T_B_prime_W[:3, :3]
    B_prime_trans = T_B_prime_W[:3, 3]

    return B_prime_rot, B_prime_trans

def euler_to_quaternion(euler_angles, degrees=True):
    """
    將尤拉角轉換成四元數
    :param euler_angles: (rx, ry, rz) 三個旋轉角度 (以弧度或度為單位)
    :param degrees: 是否以度數為單位 (預設為True)
    :return: 四元數 (qx, qy, qz, qw)
    """
    # 使用 SciPy 進行轉換
    r = R.from_euler('xyz', euler_angles, degrees=degrees)
    quaternion = r.as_quat()  # 輸出格式為 [qx, qy, qz, qw]
    return quaternion

def quaternion_to_matrix(q):
    """將四元數轉換為旋轉矩陣"""
    return R.from_quat(q).as_matrix()

def matrix_to_quaternion(R_mat):
    """將旋轉矩陣轉換為四元數"""
    return R.from_matrix(R_mat).as_quat()

def axis_angle_to_quaternion(axis, angle):
    """
    將軸角表示法 (axis, angle) 轉換為四元數。
    :param axis: 旋轉軸 (3D 向量)
    :param angle: 旋轉角度 (弧度)
    :return: 四元數 [x, y, z, w]
    """
    axis = np.array(axis) / np.linalg.norm(axis)  # 確保軸是單位向量
    quaternion = R.from_rotvec(axis * angle).as_quat()
    return quaternion

# --------------------------------------------------------------------------
def quaternion_to_axis_angle(quaternion):
    """
    將四元數轉換為Axis-Angle格式 (Webots適用)
    
    :param quaternion: 四元數 (x, y, z, w)
    :return: (axis_x, axis_y, axis_z, angle) -> Webots旋轉格式
    """
    # 建立Rotation物件
    rotation = R.from_quat(quaternion)
    
    # 獲取旋轉向量(rotvec)，rotvec = axis * angle
    rotvec = rotation.as_rotvec()
    
    # 計算角度
    angle = np.linalg.norm(rotvec)
    
    if angle == 0:
        # 沒有旋轉，預設軸為x軸
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = rotvec / angle  # 正規化成單位向量
    
    return axis[0], axis[1], axis[2], angle

def transform_to_world(R_A_in_W, p_A_in_W, R_in_A, p_in_A):
    """
    將在A座標系下的平移與旋轉，轉換到世界座標。

    :param R_A_in_W: A相對世界的旋轉矩陣 (3x3)
    :param p_A_in_W: A相對世界的平移向量 (3,)
    :param R_in_A: 目標物體在A下的旋轉矩陣 (3x3)
    :param p_in_A: 目標物體在A下的平移向量 (3,)
    :return: (R_in_W, p_in_W) 在世界座標下的旋轉和平移
    """
    R_in_W = R_A_in_W @ R_in_A
    p_in_W = R_A_in_W @ p_in_A + p_A_in_W
    return R_in_W, p_in_W

def world_to_local(R_A,T_A,R_B,T_B):

    # 平移轉換
    T_A_in_B = np.linalg.inv(R_B) @ (T_A - T_B)

    # 旋轉轉換
    R_A_in_B = np.linalg.inv(R_B) @ R_A

    return R_A_in_B,T_A_in_B

def axis_angle_to_rotation_matrix(axis, angle):
    """
    將 axis-angle 轉換為旋轉矩陣。

    參數:
    - axis: (list 或 array) [x, y, z]，旋轉軸（不必是單位向量，函數內會正規化）
    - angle: (float) 旋轉角度，單位：弧度

    回傳:
    - rotation_matrix: (3x3 numpy array) 旋轉矩陣
    """
    axis = np.array(axis)
    if np.linalg.norm(axis) == 0:
        raise ValueError("Rotation axis cannot be zero vector.")
    axis = axis / np.linalg.norm(axis)  # 正規化

    rotation = R.from_rotvec(axis * angle)
    return rotation.as_matrix()



# --------------------------------------------------------------------------

def compute_target_end_effector_pose(p_w, q_w, p_e, q_e, p_w_new, q_w_new):
        """
        根據工件的初始和目標位姿計算新的末端執行器位姿。
        輸入工件與末端執行器的初始位置以確定兩坐標系之間的相對位置，接著就可以根據工件的目標位置推算出相對應末端執行器的位置
        :param p_w: 初始工件位置 (x, y, z)
        :param q_w: 初始工件四元數 (x, y, z, w)
        :param p_e: 初始末端執行器位置 (x, y, z)
        :param q_e: 初始末端執行器四元數 (x, y, z, w)
        :param p_w_new: 新的工件位置 (x, y, z)
        :param q_w_new: 新的工件四元數 (x, y, z, w)
        :return: (p_e_new, q_e_new) 新的末端執行器位置和四元數
        """
        # 將四元數轉換為旋轉矩陣
        R_w = quaternion_to_matrix(q_w)
        R_e = quaternion_to_matrix(q_e)
        R_w_new = quaternion_to_matrix(q_w_new)
        
        # 轉換為齊次變換矩陣
        T_w = np.eye(4)
        T_w[:3, :3] = R_w
        T_w[:3, 3] = p_w
        
        T_e = np.eye(4)
        T_e[:3, :3] = R_e
        T_e[:3, 3] = p_e
        
        # 計算工件相對於末端執行器的變換矩陣
        T_we = np.linalg.inv(T_w) @ T_e
        
        # 計算新的工件變換矩陣
        T_w_new = np.eye(4)
        T_w_new[:3, :3] = R_w_new
        T_w_new[:3, 3] = p_w_new
        
        # 計算新的末端執行器變換矩陣
        T_e_new = T_w_new @ T_we
        
        # 提取新的位置與旋轉矩陣
        p_e_new = T_e_new[:3, 3]
        R_e_new = T_e_new[:3, :3]
        
        # 轉換為四元數
        q_e_new = matrix_to_quaternion(R_e_new)
        
        return p_e_new, q_e_new


def directly_go_to_target(quaternion,p_w_new): #no angle
    """
    輸入工件的目標位置以及目標角度(以軸角表示法表示),以及手臂的初始姿態
    p_w_new:工件目標位置
    axis:旋轉軸
    angle:旋轉角
    robot_initial_pos:手臂各軸的初始角度 [a1,a2,a3,a4,a5,a6]
    quaternion:工件的目標角度(以四位元數表示)
    """
    #輸入工件目標位置的旋轉(軸角表示法)，將其轉換為四位元數
    # axis = [0.654636,0.377894,-0.654712]
    # angle = 2.41868
    #輸入工件目標位置的平移與旋轉()
    # p_w_new = np.array([0.059, 0.646998, 0.46])
    q_w_new = quaternion
    #輸入工件與末端執行器初始位置(四位元數表示旋轉)，根據工件目標位置計算末端執行器的目標位置
    p_w = np.array([0.527245,-0.00104326,0.6771]) #工件平面new
    q_ = np.array([-0.575961,-0.577646,-0.578442,2.09572]) #工件平面new
    q_w = axis_angle_to_quaternion(q_[:3],q_[3])
    print("工件座標",p_w)
    print("工件座標q",q_w)
    pos=get_endpoint_position([0,0,0,0,0,0,0])
    p_e = np.array([pos[0][3],pos[1][3],pos[2][3]])#手臂末端位置
    print("1=",p_e)
    p_e = np.array([0.424132,-0.00133899,0.67624])
    print("2=",p_e)
    q_ = np.array([0.999998, -0.00169276,  0.00125543,  1.57158])#手臂末端四位元
    q_e = axis_angle_to_quaternion(q_[:3],q_[3])
    print("手臂座標",pos[0][3],pos[1][3],pos[2][3])
    print("手臂座標q",q_e)
    p_e_new, q_e_new = compute_target_end_effector_pose(p_w, normalize_quaternion(q_w), p_e, normalize_quaternion(q_e), p_w_new, normalize_quaternion(q_w_new))#末端執行器的目標位置
    # R_e_new = quaternion_to_matrix(q_e_new)#將四位元數轉換成旋轉矩陣
    # R_e_new= fix_webots_frame_error(R_e_new)#由於urdf的末端執行器坐標系與webots裡面不同，進行修正
    # ikAnglesD=get_IK_angle(p_e_new,R_e_new)#利用逆向運動學求出機器手臂各軸角度

    #,ikAnglesD
    return p_e_new,q_e_new

def normalize_quaternion(q):
    return q / np.linalg.norm(q)
# #-----------------------------------------------------------------------

# import numpy as np
# from scipy.spatial.transform import Rotation as R

motors = []
motors.append(supervisor.getDevice('J1'))
motors.append(supervisor.getDevice('J2'))
motors.append(supervisor.getDevice('J3'))
motors.append(supervisor.getDevice('J4'))
motors.append(supervisor.getDevice('J5'))
motors.append(supervisor.getDevice('J6'))

# # 初始化手臂位置
# for motor in motors:
#     motor.setPosition(0.0)

# sensors = []
# for motor in motors:
#     sensor = motor.getPositionSensor()
#     sensor.enable(timestep)
#     sensors.append(sensor)

# # 初始化關節角度
# sensor_values = [s.getValue() for s in sensors]
# current_joint_angles = [0] + sensor_values  # ikpy 需要7個joint (base + 6 joint)

# # 測試用點
# target_pos = [-0.011807, -0.470684, 0.647423]  # 目標位置
# target_quat = [-0.353538, 0.612378, -0.35356, 0.612372]  # 目標四元數
# target_rot_matrix = quaternion_to_matrix(target_quat)  # 轉成旋轉矩陣

# # 使用IK求解角度
# ik_angles = get_IK_angle(target_pos, target_rot_matrix, starting_nodes_angles=current_joint_angles)
# print("IK Angles:", ik_angles[1:])

# # 讓馬達動到目標位置
# for n, motor in enumerate(motors):
#     motor.setPosition(ik_angles[n + 1])  # ikpy第0個是虛擬base
# step_counter=0
# # 確認末端點位置
# while supervisor.step(timestep) != -1:
#     sensor_values = [s.getValue() for s in sensors]
#     current_joint_angles = [0] + sensor_values
#     now_pos = get_endpoint_position(current_joint_angles)
#     print("Current End-Effector Position:", now_pos[0][3], now_pos[1][3], now_pos[2][3])

#     node = supervisor.getFromDef("solidd")
#     node.getField('translation').setSFVec3f([now_pos[0][3], now_pos[1][3], now_pos[2][3]])

#     step_counter += 1
#     if step_counter > 500:
#         break

# from spatialmath import SE3
# import numpy as np

# def get_fk(joint_angles):
#     q1, q2, q3, q4, q5, q6 = joint_angles

#     # Base座標偏移 (Webots中的全域偏移)
#     T_base = SE3(-0.005606, 3.81657e-05, -0.018816)

#     # 各關節的DH轉換 (根據URDF + VRML確認)
#     T1 = SE3(0, 0, 0.042741) * SE3.Rz(q1)
#     T2 = SE3(0.05, 0, 0.28726) * SE3.Ry(q2)
#     T3 = SE3(0, 0, 0.33) * SE3.Ry(-q3)  # 注意Webots是(0,-1,0)
#     T4 = SE3(0.088001, 0, 0.035027) * SE3.Rx(-q4)
#     T5 = SE3(0.2454, 0, 0) * SE3.Ry(-q5)
#     T6 = SE3(0.05, 0, 0) * SE3.Rx(-q6)

#     # 最後J6有一個固定的X軸90度旋轉（Webots的rotation）
#     T_tool = SE3.Rx(np.pi/2)

#     # 完整的正向運動學
#     T = T_base * T1 * T2 * T3 * T4 * T5 * T6 * T_tool

#     return T

# # 例子：假設6個角度如下（單位: rad）
# joint_angle_list = [0, 0, 0, 0, 0, 0]

# T = get_fk(joint_angle_list)

# print("End Effector Position:", T.t)  # 平移向量 (X,Y,Z)
# print("End Effector Rotation Matrix:\n", T.R)  # 旋轉矩陣

import numpy as np


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ])


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

# joint_angles = [0.5, 0, 0, 0, 0, 0]  # 所有關節角度 (單位: 弧度)
_=0
__=0
nn=0
for n, motor in enumerate(motors):
    motor.setPosition(0)
angles=[0,0,0,0,0,0,0]
# for n, motor in enumerate(motors):
#     motor.setPosition(angles[n+1])
# base_node = supervisor.getFromDef("LRMate200iD")
# base_node.getField('translation').setSFVec3f([0,0,0])
# base_node = supervisor.getFromDef("J1")
# base_node.getField('translation').setSFVec3f([0,0,0.169])

while supervisor.step(timestep) != -1:
    for n, motor in enumerate(motors):
        motor.setPosition(angles[n+1])
    if nn==7:
        break 
    angles[nn+1]=_*0.1-1
    if _>=20:
      nn+=1
      angles[nn]=0
      _=0
      
       
#     base_node = supervisor.getFromDef("LRMate200iD")
#     base_node.getField('translation').setSFVec3f([0.0, -0.0804225935, 0.0])
    # base_node = supervisor.getFromDef("solid")
    # solid_pos=base_node.getField('translation').getSFVec3f()
    # solid_rot=base_node.getField('rotation').getSFVec3f()
    
    j6_node = supervisor.getFromDef("J6")
    j6_pos=j6_node.getPosition()
    j6_rota=j6_node.getOrientation()

    # print("pose_pos",pose_pos)
    # print("pose_rota",euler_to_axis_angle(pose_rota_[0],pose_rota_[1],pose_rota_[2]))
    # base_node = supervisor.getFromDef("LRMate200iD")
    # cal_pos=get_endpoint_position(angles)
    
    
    # R_B=np.array(axis_angle_to_rotation_matrix(solid_rot[:3],solid_rot[3])).reshape(3,3)
    # T_B=np.array(solid_pos)

    # angless=get_IK_angle(T_B,R_B,"all",[0,0,0,0,0,0,0])
    # for n, motor in enumerate(motors):
    #     motor.setPosition(angless[n+1])
    cal_pos=get_endpoint_position(angles)

    # j6_node = supervisor.getFromDef("J6")
    # j6_pos=j6_node.getPosition()
    
    __+=1
    if __ >30:
        __=0
        _+=1
        # print("angles",angles)
        # print("angless",angless)
        # print("j_pos",j6_pos)
        # print("solid_pos",solid_pos)
        # print("solid_rot",solid_rot)
        # print("cal_pos",cal_pos[0][3],cal_pos[1][3],cal_pos[2][3])
        # print("error_solid",solid_pos-np.array([cal_pos[0][3],cal_pos[1][3],cal_pos[2][3]]))
        # print("j6_pos",j6_pos)
        error=j6_pos-np.array([cal_pos[0][3],cal_pos[1][3],cal_pos[2][3]])
        if abs(error[0])>0.0003 or abs(error[1])>0.0003 or abs(error[2])>0.0003:
            print("error_j6",error)
            print("cal_pos",cal_pos[0][3],cal_pos[1][3],cal_pos[2][3])
            print("j6_pos",j6_pos)
        # print("gpt_cal_pos",gpt_cal_pos)
        # print("-->w_pose_t",w_pose_t)
    #     _=0
    #     __+=1
    # if __==6:
        # break



# print("w_a_r",w_a_r)
# print("End Effector Position:", pos)
# print("End Effector Rotation Matrix:\n", rot)

# poss=get_endpoint_position([0]+joint_angles)
# print("End Effector Position:", poss[0][3],poss[1][3],poss[2][3])
# print("End Effector Rotation Matrix:\n", poss[:3,:3])
# base_translation = np.array([-0.005606, 3.81656697e-05, -0.018816])  # Webots 的 Robot 初始位置

# poss = get_endpoint_position([0]+joint_angles)
# poss[:3, 3] += base_translation  # 位置修正

# print("End Effector Position after correction:", poss[0,3], poss[1,3], poss[2,3])
