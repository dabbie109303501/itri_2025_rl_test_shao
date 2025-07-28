import numpy as np
from controller import Robot, Supervisor,Connector
from ikpy.chain import Chain
import pandas as pd
import numpy as np
from gymnasium import Env
from gymnasium import spaces
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.transform import Rotation as R #如果需要將旋轉矩陣轉換為 四元數 (quaternion) 或 歐拉角 (Euler angles)，可以使用
from collections import defaultdict
import pandas as pd
import random

# 全域或類別屬性都可以
# supervisor = Supervisor()


#利用順項運動學計算出末端軸位置
def get_endpoint_position(angles):#input為機器手臂各軸馬達的位置(角度，以弧度表示)
    endpoint_position=robot_chain.forward_kinematics(angles)
    return endpoint_position #output為手臂末端軸位置(以轉至矩陣表示)

def rotation_matrix_to_euler_angles(R):#將旋轉矩陣轉換為歐拉角
    """
    輸入為3x3的旋轉矩陣,輸出歐拉角表示法的rx,ry,rz
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

def calculate_A_prime(R_BA, t_BA, R_B_prime, t_B_prime):#輸入工件路徑採樣點研磨時的座標平移與歐拉角，輸出工件座標研磨時的平移與歐拉角
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
    return R_A_prime, t_A_prime#尤拉角的單位應該為度

def Rotation_matrix(rx,ry,rz):#將歐拉角轉換為旋轉矩陣
    '''
    將歐拉角轉換為旋轉矩陣
    rx,ry,rz為歐拉角

    '''
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


def euler_to_axis_angle(rx, ry, rz):#欧拉角轉换为轴-角表示(webots內的rotation以轴-角表示)
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

def quaternion_angle(q1, q2):
    '''
    計算兩個四元數之間旋轉角度 𝜃
    qi = np.array([qx, qy, qz, qw])
    '''
    # 將四元數正規化 (確保它們為單位四元數)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # 計算內積 (dot product)
    dot_product = np.dot(q1, q2)

    # 修正數值誤差，確保內積在 [-1, 1] 範圍內
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # 計算旋轉角度 (θ = 2 * arccos(|q1 ⋅ q2|))
    theta = 2 * np.arccos(abs(dot_product))

    # 轉換為度數並回傳
    return np.degrees(theta)

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

def quaternion_to_matrix(q):
    """將四元數轉換為旋轉矩陣"""
    return R.from_quat(q).as_matrix()

def matrix_to_quaternion(R_mat):
    """將旋轉矩陣轉換為四元數"""
    return R.from_matrix(R_mat).as_quat()
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

def fix_webots_frame_error(Rotation_matrix):
    # 假設給定 A 座標系相對於世界座標的旋轉矩陣 R
    '''
    由於urdf裡面的末端執行器坐標系與webots裡面的末端執行器坐標系不太一樣,
    先將webots裡面末端執行器的目標角度(以相對世界坐標系的旋轉矩陣表示)轉換成urdf裡面的目標角度,才可以接下來的逆向運動學計算。
    R = np.array([[0, -1, 0], 
                [1,  0, 0], 
                [0,  0, 1]])  # 示例旋轉矩陣
    '''

    # 沿著 A 座標系的 x 軸旋轉 -90 度
    Rx_A = R.from_euler('x', -90, degrees=True).as_matrix()

    # 計算新的旋轉矩陣 R' = R @ Rx_A
    R_prime = Rotation_matrix @ Rx_A
    return R_prime
def get_IK_angle(target_position, target_orientation,initial_position, orientation_axis="all"):
    """
    计算逆向运动学角度
    :param target_position: 目标末端位置 [x, y, z]
    :param target_orientation: 目标方向 (3x3 旋转矩阵)
    :param orientation_axis: 指定对齐轴 ("x", "y", "z")，或者 "all" 进行完整姿态匹配
    :return: 6 轴角度 (弧度制)
    :initial_position: 手臂各軸的初始角度 [a1,a2,a3,a4,a5,a6]
    """
    # Initial_Position = np.zeros(10) ###
    # Initial_Position[2:8]=initial_position ###

    # 1. 先複製一份 mask，並排除 Base link（index 0）
    mask = robot_chain.active_links_mask.copy()
    mask[0] = False

    # 2. 建一個與 mask 同長度的零向量
    init_full = np.zeros(mask.shape[0])
    # 3. 只在 mask==True 的位置，填入你的 6 軸 initial_position
    init_full[mask] = initial_position


    # 计算逆运动学
    ik_angles = robot_chain.inverse_kinematics(
        target_position,
        target_orientation=target_orientation ,
        orientation_mode=orientation_axis,
        initial_position=init_full
    )
    # return ik_angles[2:8] # 取 6 轴角度 (去掉基座)
    # 4. **穩健地**用 mask 把 active joints 抽出來
    target_angles = ik_angles[mask]

    # 5. 確認數量正確（可選）
    assert target_angles.shape[0] == len(initial_position), \
        f"IK 回傳 {target_angles.shape[0]} 軸，但環境有 {len(initial_position)} 軸"

    return target_angles

# def directly_go_to_target(quaternion,p_w_new,robot_initial_pos):
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
    q_w_new = np.array(quaternion)
    #輸入工件與末端執行器初始位置(四位元數表示旋轉)，根據工件目標位置計算末端執行器的目標位置
    # p_w = np.array([-0.488223, 0.000823146, 0.000823146])
    # q_w = np.array([ 0.0, -1,  0.0,  1.21326795e-04])  # 單位四元數
    # p_e = np.array([0.816992, 0.233936, 0.0628227])
    # q_e = np.array([ 0.0, -1,  0.0,  1.21326795e-04])


    p_w = np.array([-0.488223, 0.000823146, 0.000823146])
    # q_w = axis_angle_to_quaternion(-0.57627, 0.578229, 0.57755, 2.09631)
    # print("q_w:",q_w)
    q_w = np.array([0.499171, -0.499340, 0.501038, 0.500449])
    p_e = np.array([-0.425723, 0.000796888, 0.682031])
    # q_e = axis_angle_to_quaternion(-0.00120166, -0.707374, -0.706839, 3.13978)
    q_e= np.array([  0.00090633,  -0.00120166,  -0.70737343,  -0.70683843  ])
    
    p_e_new, q_e_new = compute_target_end_effector_pose(p_w, q_w, p_e, q_e, p_w_new, q_w_new)#末端執行器的目標位置

    R_e_new = quaternion_to_matrix(q_e_new)#將四位元數轉換成旋轉矩陣
    R_e_new=fix_webots_frame_error(R_e_new)#由於urdf的末端執行器坐標系與webots裡面不同，進行修正
    ikAnglesD=get_IK_angle(p_e_new,R_e_new,robot_initial_pos)#利用逆向運動學求出機器手臂各軸角度

    return ikAnglesD

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

def quaternion_to_axis_angle(q):
    x, y, z, w = q
    angle = 2 * np.arccos(w)
    s = np.sqrt(1 - w*w)
    if s < 1e-6:
        return [1, 0, 0, 0]  # 無旋轉
    return [x/s, y/s, z/s, angle]

class World(Env):
    def __init__(self, ideal_feed: float = 0.0001, target_force=2.0):

        super().__init__()
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())

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
        for i in range(len(self.df)-1,-1,-1):
            num_plane.append(self.df[i][0])
            num_path.append(self.df[i][1])
            xyz.append([self.df[i][2]/1000,self.df[i][3]/1000,self.df[i][4]/1000])
            rxryrz.append([self.df[i][5],self.df[i][6],self.df[i][7]])
            
        self.samplept_r=[]; self.samplept_t=[]

        for point_num in range(len(self.df)): #len(df)
            rel_r_samplept=Rotation_matrix(
                rxryrz[point_num][0],rxryrz[point_num][1],rxryrz[point_num][2])
            rel_t_samplept=np.array(xyz[point_num])
            abs_r_contactpt_frame=Rotation_matrix(0,120,0)
            abs_t_contactpt_frame=np.array([0.53-0.06*math.sqrt(3)/2+0.001, 0, 0.775+0.06/2-0.001]) #設定的與砂帶接觸點
            abs_samplept_r, abs_samplept_t = calculate_A_prime(
                rel_r_samplept, rel_t_samplept, abs_r_contactpt_frame, abs_t_contactpt_frame)
            self.samplept_r.append(abs_samplept_r)
            self.samplept_t.append(abs_samplept_t)


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

        #-------------------------------------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        # 重置模擬／工件位置
        self.supervisor.simulationReset()
        self.idx = 0
        self.feed = self.ideal_feed

        # 對準第一個取樣點，並施加初始進給量
        # x,y,z = self.path_pts[0]

        # 假設以 Z 軸負方向推進工件
        pathptnode=self.supervisor.getFromDef(str('p'+str(0)))
        pathpt_tran=pathptnode.getField('translation')
        pathpt_rota=pathptnode.getField('rotation')
        pathpt_tran.setSFVec3f([self.samplept_t[self.idx][0]+self.feed*math.sqrt(3)/2, self.samplept_t[self.idx][1], self.samplept_t[self.idx][2]-self.feed/2])
        x,y,z,a=euler_to_axis_angle(self.samplept_r[self.idx][0], self.samplept_r[self.idx][1], self.samplept_r[self.idx][2])
        pathpt_rota.setSFRotation([float(x), float(y), float(z), float(a)])

        # 推一個 timestep，讓感測器生效
        self.supervisor.step(self.timestep)

        # 量測初始力並回傳狀態
        # F0 = self.force_sensor.getValue()
        # state = np.array([F0, self.target_force - F0], dtype=np.float32)

        fx, fy, fz = self.force_sensor.getValues()[:3]         # 讀取三軸力
        print("fx={:.3f},fy={:.3f},fz={:.3f}".format(fx, fy, fz))
        # data.append([t, fx, fy, fz])
        F0 = fx  # 假設 Y 軸為主要力方向
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
        
        
        pathptnode=self.supervisor.getFromDef(str('p'+str(0)))
        pathpt_tran=pathptnode.getField('translation')
        pathpt_rota=pathptnode.getField('rotation')
        pathpt_tran.setSFVec3f([self.samplept_t[self.idx][0]+self.feed*math.sqrt(3)/2, self.samplept_t[self.idx][1], self.samplept_t[self.idx][2]-self.feed/2])
        x,y,z,a=euler_to_axis_angle(self.samplept_r[self.idx][0], self.samplept_r[self.idx][1], self.samplept_r[self.idx][2])
        pathpt_rota.setSFRotation([float(x), float(y), float(z), float(a)])


        # base_point = np.array(self.samplept_t[self.current_index])
        # target_pos = base_point + np.array([0, 0, delta_d])
        # target_q = self.samplept_r[self.current_index]
        
        # 
        self.supervisor.step(self.timestep)

        # 讀力、計算狀態與回饋
        # f = self.force_sensor.getValue()
        # print("force:", f)
        # e = self.target_force - f
        # state = np.array([f, e], dtype=np.float32)

        fx, fy, fz = self.force_sensor.getValues()[:3]         # 讀取三軸力
        print("fx={:.3f},fy={:.3f},fz={:.3f}".format(fx, fy, fz))
        e = self.target_force - fx
        state = np.array([fx, e], dtype=np.float32)

        # 獎勵回饋：誤差越小越好，可酌量加入動作懲罰
        reward = -abs(e)
        
        # # 5. 讀力、計算 state & reward
        # fx, fy, fz = self.force_sensor.getValues()[:3]
        # F, err = fz, self.target_force - fz
        # state = np.array([F, err], dtype=np.float32)
        # reward = -abs(err) - 0.01 * abs(delta_d)
        
        # # 6. 更新 index
        # self.current_index += 1
        # t += timestep / 1000.0  # 更新時間 (毫秒)
        # done = self.current_index >= len(self.samplept_t)

        done      = (self.idx + 1 >= len(self.samplept_t))
        truncated = False
        info      = {}

        return state, reward, done, truncated, info

    # def step(self, action):
    #     # 對工具施加深度變化
    #     d = float(action[0])
    #     # 讀取目前工具位置並更新
    #     pos = np.array(self.tool_node.getPosition())
    #     pos[2] += d  # 假設 Z 方向為研磨深度
    #     self.tool_node.setPosition(pos.tolist())

    #     # 進入下一個 time step
    #     self.supervisor.step(self.supervisor.getBasicTimeStep())

    #     # 量測力與計算誤差
    #     F = self.force_sensor.getValue()
    #     err = self.target_force - F
    #     state = np.array([F, err], dtype=np.float32)

    #     # 計算回饋
    #     reward = -abs(err) - 0.01 * abs(d)

    #     # 更新路徑指標
    #     done = False
    #     self.current_index += 1
    #     if self.current_index >= len(self.path_points):
    #         done = True

    #     return state, reward, done, {}

    # def _load_path(self):
    #     # TODO: 載入預先定義的研磨路徑點陣列
    #     return [(x, y, z) for ... in ...]

    def render(self, mode='human'):
        pass

    











