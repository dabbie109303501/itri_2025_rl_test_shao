import numpy as np
from controller import Supervisor
from ikpy.chain import Chain
import pandas as pd
from gymnasium import Env
from gymnasium import spaces
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.transform import Rotation as R #如果需要將旋轉矩陣轉換為 四元數 (quaternion) 或 歐拉角 (Euler angles)，可以使用
from collections import defaultdict
import random
import matplotlib.pyplot as plt

import warnings

# 抑制 IKPy 的固定 Base 警告
warnings.filterwarnings("ignore", category=UserWarning, module="ikpy.chain")
# 抑制 Gymnasium Box precision 警告
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")

robot_chain = Chain.from_urdf_file("LRMate-200iD_try_fix_arm.urdf",base_elements=['Base'])

#利用順項運動學計算出末端軸位置
def get_endpoint_position(angles):
    endpoint_position=robot_chain.forward_kinematics(angles)
    return endpoint_position

#利用逆向運動學，根據末端軸位置以及角度推算手臂個軸角度
def get_IK_angle(target_position, target_orientation, orientation_axis="all",starting_nodes_angles=[0,0,0,0,0,0,0]): 
    # 初始化機器人鏈條
    ikAnglesD= robot_chain.inverse_kinematics(
    target_position,
    target_orientation=target_orientation,
    orientation_mode=orientation_axis,
    initial_position=starting_nodes_angles,
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
    q_w_new = quaternion
    #輸入工件與末端執行器初始位置(四位元數表示旋轉)，根據工件目標位置計算末端執行器的目標位置
    p_w = np.array([0.527245,-0.00104326,0.6771]) #工件平面new
    q_ = np.array([-0.575961,-0.577646,-0.578442,2.09572]) #工件平面new
    q_w = axis_angle_to_quaternion(q_[:3],q_[3])
    # print("工件座標",p_w)
    # print("工件座標q",q_w)
    pos=get_endpoint_position([0,0,0,0,0,0,0])
    p_e = np.array([pos[0][3],pos[1][3],pos[2][3]])#手臂末端位置
    p_e = np.array([0.424132,-0.00133899,0.67624])
    q_ = np.array([0.999998, -0.00169276,  0.00125543,  1.57158])#手臂末端四位元
    q_e = axis_angle_to_quaternion(q_[:3],q_[3])
    p_e_new, q_e_new = compute_target_end_effector_pose(p_w, normalize_quaternion(q_w), p_e, normalize_quaternion(q_e), p_w_new, normalize_quaternion(q_w_new))#末端執行器的目標位置

    return p_e_new,q_e_new

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

#-----------------------------------------------------------------------


class World(Env):
    def __init__(self, ideal_feed: float = 0.000685, target_force=30.0):

        super().__init__()
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())

        self.robot_chain = Chain.from_urdf_file("LRMate-200iD_try_fix_arm.urdf",base_elements=['Base'])

        self.approach_dist = 0.2  # 2 cm 的安全距離

        # 馬達與感測器
        self.motors = [self.supervisor.getDevice(f'J{i+1}') for i in range(6)]
        self.sensors = []
        for m in self.motors:
            ps = m.getPositionSensor()
            ps.enable(self.timestep)
            self.sensors.append(ps)

        #力感測器
        self.target_force = target_force
        self.force_sensor = self.supervisor.getDevice('force sensor')
        self.force_sensor.enable(self.timestep)

        # 讀取路徑
        file_path = "./paths/flat_transformed.csv"
        # file_path = "./paths/single_curved_transformed.csv"
        # file_path = "./paths/double_curved_transformed.csv"
        self.df = np.array(pd.read_csv(file_path, header=None))
        num_plane=[];num_path=[];xyz=[];rxryrz=[]
        for i in range(len(self.df)):
            num_plane.append(self.df[i][0])
            num_path.append(self.df[i][1])
            xyz.append([self.df[i][2]/1000,self.df[i][3]/1000,self.df[i][4]/1000])
            rxryrz.append([self.df[i][5],self.df[i][6],self.df[i][7]])
            
        self.samplept_r=[]; self.samplept_t=[]
        self.apaths=[]; self.aqu=[]

        for point_num in range(len(self.df)): #len(df)
            rel_r_samplept=Rotation_matrix(
                rxryrz[point_num][0],rxryrz[point_num][1],rxryrz[point_num][2])
            rel_t_samplept=np.array(xyz[point_num])
            # abs_r_contactpt_frame=Rotation_matrix(0,120,0)
            abs_r_contactpt_frame=Rotation_matrix(180,-60,0)
            # abs_t_contactpt_frame=np.array([0.53-0.06*math.sqrt(3)/2+0.001, 0, 0.775+0.06/2-0.001]) #設定的與砂帶接觸點
            abs_t_contactpt_frame=np.array([0.67-0.06*math.sqrt(3)/2+0.001, 0, 0.62+0.06/2-0.001]) #設定的與砂帶接觸點
            abs_samplept_r, abs_samplept_t = calculate_A_prime(
                rel_r_samplept, rel_t_samplept, abs_r_contactpt_frame, abs_t_contactpt_frame)
            self.samplept_r.append(abs_samplept_r)
            self.samplept_t.append(abs_samplept_t)
            abs_samplept_r_q=euler_to_quaternion(abs_samplept_r)
            p,q=directly_go_to_target(abs_samplept_r_q,abs_samplept_t)
            self.apaths.append(p)
            self.aqu.append(q)

        self.N = len(self.apaths)

        # 初始進給量與目標力度
        self.ideal_feed = ideal_feed
        self.target_force = target_force

        # 觀察空間：當前力、誤差
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf]),
            high=np.array([np.inf, np.inf]),
            dtype=np.float32
        )
        # 動作空間：深度變化量 (mm)
        self.action_space = spaces.Box(
            low=np.array([-0.001]), high=np.array([0.001]), dtype=np.float32
        )

        self.rewards=[]
        self.count_epsiode=0
        self.reward=[]
        self.avg_reward=[]

        #-------------------------------------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if len(self.rewards)!=0:
            self.avg_reward.append(sum(self.reward)/len(self.reward))
            self.count_epsiode+=1
            if self.count_epsiode==20:
                x1=[_ for _ in range(len(self.rewards))]
                y1=[_ for _ in self.rewards]
                plt.plot(x1,y1)
                plt.savefig("./episode/-0.1abs(e)-1abs(delta_d)_20.png")
                plt.show()
                x2=[_ for _ in range(len(self.avg_reward))]
                y2=[_ for _ in self.avg_reward]
                plt.plot(x2,y2)
                plt.savefig("./episode/-0.1abs(e)-1abs(delta_d)_avg_reward_20.png")
                plt.show()
        # SB3 / Gymnasium 兼容
        if seed is not None:
            np.random.seed(seed)

        # 重置模擬／工件位置
        self.supervisor.simulationReset()
        self.idx = 0
        self.feed = self.ideal_feed

        # 初始對準：用 IK 計算並下發到各軸
        target_pos = self.apaths[0] + np.array([self.feed*math.sqrt(3)/2, 0, -self.feed/2])
        target_ori = quaternion_to_matrix(self.aqu[0])

        # 避免碰撞的安全點

        # print("apaths[0]:", self.apaths[0])
        safety_pos =np.array([0.514, 0.008, 0.768])

        # IK 到 safety pose
        ik_approach = get_IK_angle(
            safety_pos,
            target_ori,
            starting_nodes_angles=[0] + [s.getValue() for s in self.sensors]
        )
        # 下發到馬達
        # for i, m in enumerate(self.motors):
        #     m.setPosition(ik_approach[i+1])
        #     print(f"Motor {i+1} set to angle: {ik_approach[i+1]}")

        # 讓模擬跑幾個 step，好讓手臂移動到位
        for _ in range(5):
            for i, m in enumerate(self.motors):
                m.setPosition(ik_approach[i+1])
                # print(f"Motor {i+1} set to angle: {ik_approach[i+1]}")
            self.supervisor.step(self.timestep)

        # 末端對準到第一個路徑點
        joint_angles = get_IK_angle(
            target_pos, target_ori,
            starting_nodes_angles=[0] + [s.getValue() for s in self.sensors]
            
        )

        for _ in range(25):
            self.supervisor.step(self.timestep)
            for i, m in enumerate(self.motors):
                m.setPosition(joint_angles[i+1])
                # print(f"Motor {i+1} set to angle: {joint_angles[i+1]}")
            self.supervisor.step(self.timestep)

        # for i, m in enumerate(self.motors):
        #     m.setPosition(joint_angles[i+1])

        # 讓模擬跑一格
        # self.supervisor.step(self.timestep)


        # 量測初始力並回傳狀態

        # 讀取單軸力
        # F0 = self.force_sensor.getValue()
        # state = np.array([F0, self.target_force - F0], dtype=np.float32)

        # 讀取三軸力
        fx, fy, fz = self.force_sensor.getValues()[:3]         
        print("fx={:.3f},fy={:.3f},fz={:.3f}".format(fx, fy, fz))
        # data.append([t, fx, fy, fz])
        # F0 = fy  # 假設 Y 軸為主要力方向
        F0 = fz  # 假設 z 軸為主要力方向
        state = np.array([F0, self.target_force - F0], dtype=np.float32)

        return state

    def step(self, action):
        # 更新進給量
        delta_d = float(action[0])
        self.feed = max(0.0, self.feed + delta_d)

        # 前進到下一個路徑點，並以新的進給量對準
        self.idx += 1
        done = (self.idx >= len(self.df))
        if done:
            # 到達終點  
            return None, 0.0, True, {}
        
        # IK → 末端位置＆姿態
        target_pos = self.apaths[self.idx] + np.array([self.feed*math.sqrt(3)/2, 0, -self.feed/2])
        target_ori = quaternion_to_matrix(self.aqu[self.idx])
        joint_angles = get_IK_angle(target_pos, target_ori, starting_nodes_angles=[0] + [s.getValue() for s in self.sensors])

        # print("target_pos:", target_pos)

        for _ in range(1):
            # self.supervisor.step(self.timestep)
            for i, m in enumerate(self.motors):
                m.setPosition(joint_angles[i+1])
                # print(f"Motor {i+1} set to angle: {joint_angles[i+1]}")
            fx, fy, fz = self.force_sensor.getValues()[:3]         # 讀取三軸力
            
            print("fx={:.3f},fy={:.3f},fz={:.3f}".format(fx, fy, fz))   
            self.supervisor.step(self.timestep)

        # motors = [self.supervisor.getDevice(f'J{i+1}') for i in range(6)]
        # print("Joint angles:", joint_angles[1:])

        # for i, m in enumerate(self.motors):
        #     m.setPosition(joint_angles[i+1])

        # 5. 讀當前各關節角
        current_angles = [s.getValue() for s in self.sensors]
        # 6. 計算並印出實際末端位置
        endpoint_tf = get_endpoint_position([0.0] + current_angles)  # 4×4 變換矩陣
        current_pos = endpoint_tf[:3, 3]                          # 取平移部分
        # print(f"[Step {self.idx}] 實際末端位置: {current_pos}")


        # 讓模擬跑一格
        # self.supervisor.step(self.timestep)

        # 讀力、計算狀態與回饋
        # 單軸
        # f = self.force_sensor.getValue()
        # print("force:", f)
        # e = self.target_force - f
        # state = np.array([f, e], dtype=np.float32)

        fx, fy, fz = self.force_sensor.getValues()[:3]         # 讀取三軸力
        # print("fx={:.3f},fy={:.3f},fz={:.3f}".format(fx, fy, fz))
        # e = self.target_force - fy
        
        F = fz  # 假設 z 軸為主要力方向
        e = self.target_force - F
        state = np.array([F, e], dtype=np.float32)

        # 獎勵回饋：誤差越小越好，可酌量加入動作懲罰
        reward = -abs(e)
        
        
        # reward = 1* (0.05-abs(e)/self.target_force)- 0.1 *  abs(delta_d)
        self.rewards.append(reward)
        self.reward.append(reward)

        done      = (self.idx + 1 >= self.N)
        truncated = False
        info      = {}

        # 讓模擬跑一格
        self.supervisor.step(self.timestep)

        return state, reward, done, truncated, info

    def render(self, mode='human'):
        pass

    











