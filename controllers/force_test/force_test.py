from controller import Supervisor,Connector
from ikpy.chain import Chain
import pandas as pd
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ikpy.chain")


supervisor = Supervisor()
# get the time step of the current world.

timestep = int(supervisor.getBasicTimeStep())


#利用順項運動學計算出末端軸位置
# def get_endpoint_position(angles):
#     endpoint_position=robot_chain.forward_kinematics(angles)
#     return endpoint_position

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
import numpy as np
from scipy.spatial.transform import Rotation as R

# 取得力感測器
force_sensor = supervisor.getDevice("force sensor")
if force_sensor is None:
    print(" ERROR: 'force_sensor' not found!")
    exit(1)
    
force_sensor.enable(timestep)  # 啟用感測器

# 準備儲存清單
data = []    # 每筆： [time(s), fx, fy, fz]
t = 0.0

# 準備儲存清單
data = []    # 每筆： [time(s), fx, fy, fz]
ffx=[]; ffy=[]; ffz=[];ft=[]
t = 0.0



file_path = "./paths/flat_transformed.csv"
# file_path = "./paths/single_curved_transformed.csv"
# file_path = "./paths/double_curved_transformed.csv"
df = np.array(pd.read_csv(file_path, header=None))
num_plane=[];num_path=[];xyz=[];rxryrz=[]
for i in range(len(df)-1,-1,-1):
    num_plane.append(df[i][0])
    num_path.append(df[i][1])
    xyz.append([df[i][2]/1000,df[i][3]/1000,df[i][4]/1000])
    rxryrz.append([df[i][5],df[i][6],df[i][7]])
    

apaths=[];aqu=[]
samplept_r=[]; samplept_t=[]
# t=[]

for point_num in range(len(df)): #len(df)
    rel_r_samplept=Rotation_matrix(
        rxryrz[point_num][0],rxryrz[point_num][1],rxryrz[point_num][2])
    rel_t_samplept=np.array(xyz[point_num])
    abs_r_contactpt_frame=Rotation_matrix(0,120,0)
    abs_t_contactpt_frame=np.array([0.53-0.06*math.sqrt(3)/2+0.00, 0, 0.775+0.06/2-0.00]) #設定的與砂帶接觸點
    abs_t_contactpt_frame=np.array([0.67-0.06*math.sqrt(3)/2+0.001, 0, 0.62+0.06/2-0.001]) #設定的與砂帶接觸點
    abs_samplept_r, abs_samplept_t = calculate_A_prime(
        rel_r_samplept, rel_t_samplept, abs_r_contactpt_frame, abs_t_contactpt_frame)
    samplept_r.append(abs_samplept_r)
    samplept_t.append(abs_samplept_t)
    # print("samplept_r",samplept_r[point_num])
    # print("samplept_t",samplept_t[point_num])




i=-5
step_counter=0
angles=[]
pre=[0,0,0,0,0,0]
while supervisor.step(timestep) != -1:
    
    if i < len(df):

        if i>0:
            pathptnode=supervisor.getFromDef(str('p'+str(0)))
            pathpt_tran=pathptnode.getField('translation')
            pathpt_rota=pathptnode.getField('rotation')
            pathpt_tran.setSFVec3f([samplept_t[i][0], samplept_t[i][1], samplept_t[i][2]])
            x,y,z,a=euler_to_axis_angle(samplept_r[i][0], samplept_r[i][1], samplept_r[i][2])
            pathpt_rota.setSFRotation([float(x), float(y), float(z), float(a)])
            # angles=get_IK_angle(apaths[i]+np.array([0.0000616,0,-0.0001067]),quaternion_to_matrix(aqu[i]),starting_nodes_angles=[0]+pre) #有加進給量
            # 三軸
            # force_values = force_sensor.getValues()  # 取得感測數據
            # print(f"End-Effector Force: X={force_values[0]:.3f}, Y={force_values[1]:.3f}, Z={force_values[2]:.3f}")
            fx, fy, fz = force_sensor.getValues()[:3]         # 讀取三軸力
            print("fx={:.3f},fy={:.3f},fz={:.3f}".format(fx, fy, fz))
            data.append([t, fx, fy, fz])

            # 單軸
            # fz = force_sensor.getValue()  # 取得感測數據
            # data.append([t, fz])
            # print(f'Fz = {fz:.3f} N')
    


                

        step_counter += 1

        if step_counter >= 5 and i<0:
            i += 1
            step_counter = 0
        elif step_counter>=1 and i>=0:
            i += 1
            step_counter = 0

    # else:
    #     for n, motor in enumerate(motors):
    #         motor.setPosition(0)
    #     break  # 所有點都移動完
    # pre=angles.copy()
    # # print(i)
    # t += timestep / 1000.0  # 更新時間 (毫秒)

print("end")