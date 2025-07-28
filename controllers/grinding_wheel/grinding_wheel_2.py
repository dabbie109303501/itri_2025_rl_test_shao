# grinding_wheel_controller.py

from controller import Supervisor
from ikpy.chain import Chain
import math
import random


sup      = Supervisor()
timestep = int(sup.getBasicTimeStep())

node = sup.getFromDef("grinding_wheel")
field = node.getField("translation")
motor_r = sup.getDevice("wheel_motor_2")

motor_x = sup.getDevice("slider_x_motor")
motor_z = sup.getDevice("slider_z_motor")


# # 定義你的中心位置 (世界座標) 與抖動幅度
# center_pos = [-0.654, -0.002, 0.613]             # # 砂輪目標位置
# jitter = [0.005, 0.0, 0.005]             # x、y、z 三軸的最大隨機偏移量
# alpha      = 0.2                   # 平滑係數 (0~1)
# prev_offset = [0.0, 0.0, 0.0]      # 上一步偏移初始化

# if motor is None:
#     print("Error: wheel_motor not found!")
#     exit(1)

# # 無限位置模式 → 持續旋轉
# motor.setPosition(float('inf'))
# motor.setVelocity(30.0)   # 30 rad/s

#--------- 參數設定 -------------
# X / Z 振動範圍 (公尺)，假設只要 ±0.005m 的微小振動
jitter_x = 0.005
jitter_z = 0.005
prev_x = 0.0  # 上一次 X 軸位置
prev_z = 0.0  # 上一次 Z 軸位置

# 設定兩個 translation motor 為位置控制模式
motor_x.setPosition(0.0)     # 初始位置設 0
motor_z.setPosition(0.0)
motor_x.setVelocity(10.0)    # 最大速度 (rad/s 或 m/s 視聯動器類型而定)
motor_z.setVelocity(10.0)

# 設定旋轉馬達為無限位置 + 指定速度
motor_r.setPosition(float('inf'))
motor_r.setVelocity(100.0)    # rad/s，砂輪轉速

#--------------------------------


# 主迴圈
while sup.step(timestep) != -1:

    # # 1. raw 隨機偏移
    # raw_dx = random.uniform(-jitter[0], jitter[0])
    # raw_dy = random.uniform(-jitter[1], jitter[1])
    # raw_dz = random.uniform(-jitter[2], jitter[2])

    # # 2. 指數平滑
    # smooth_dx = alpha * raw_dx + (1 - alpha) * prev_offset[0]
    # smooth_dy = alpha * raw_dy + (1 - alpha) * prev_offset[1]
    # smooth_dz = alpha * raw_dz + (1 - alpha) * prev_offset[2]

    # # 3. 更新 prev_offset
    # prev_offset = [smooth_dx, smooth_dy, smooth_dz]

    # # 4. 計算並設定新座標
    # new_pos = [
    #     center_pos[0] + smooth_dx,
    #     center_pos[1] + smooth_dy,
    #     center_pos[2] + smooth_dz
    # ]
    # field.setSFVec3f(new_pos)

    # ----- X 軸隨機振動 -----
    # # 在 [-jitter_x, +jitter_x] 之間取一個位置
    # target_x = random.uniform(-jitter_x, jitter_x)
    # motor_x.setPosition(target_x)

    # # ----- Z 軸隨機振動 -----
    # target_z = random.uniform(-jitter_z, jitter_z)
    # motor_z.setPosition(target_z)

    # # 如果你想要「更平滑」的振動，可以在這裡加上低通濾波：
    alpha = 0.2
    raw_x = random.uniform(-jitter_x, jitter_x)
    raw_z = random.uniform(-jitter_z, jitter_z)
    smooth_x = alpha * raw_x + (1 - alpha) * prev_x
    smooth_z = alpha * raw_z + (1 - alpha) * prev_z
    prev_x, prev_z = smooth_x, smooth_z
    motor_x.setPosition(smooth_x)
    motor_z.setPosition(smooth_z)

    pass


    

