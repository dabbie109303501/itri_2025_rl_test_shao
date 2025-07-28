# grinding_wheel_controller.py

from controller import Supervisor
from ikpy.chain import Chain
import math
import random


sup      = Supervisor()
timestep = int(sup.getBasicTimeStep())

node = sup.getFromDef("grinding_wheel")
field = node.getField("translation")
motor = sup.getDevice("wheel_motor_2")



# 定義你的中心位置 (世界座標) 與抖動幅度
center_pos = [0.674, -0.002, 0.613]             # # 砂輪目標位置
jitter = [0.0005, 0.0005, 0.0005]             # x、y、z 三軸的最大隨機偏移量
alpha      = 0.2                   # 平滑係數 (0~1)
prev_offset = [0.0, 0.0, 0.0]      # 上一步偏移初始化

if motor is None:
    print("Error: wheel_motor not found!")
    exit(1)

# 無限位置模式 → 持續旋轉
motor.setPosition(float('inf'))
motor.setVelocity(30.0)   # 30 rad/s


# 主迴圈
while sup.step(timestep) != -1:

    # 1. raw 隨機偏移
    raw_dx = random.uniform(-jitter[0], jitter[0])
    raw_dy = random.uniform(-jitter[1], jitter[1])
    raw_dz = random.uniform(-jitter[2], jitter[2])

    # 2. 指數平滑
    smooth_dx = alpha * raw_dx + (1 - alpha) * prev_offset[0]
    smooth_dy = alpha * raw_dy + (1 - alpha) * prev_offset[1]
    smooth_dz = alpha * raw_dz + (1 - alpha) * prev_offset[2]

    # 3. 更新 prev_offset
    prev_offset = [smooth_dx, smooth_dy, smooth_dz]

    # 4. 計算並設定新座標
    new_pos = [
        center_pos[0] + smooth_dx,
        center_pos[1] + smooth_dy,
        center_pos[2] + smooth_dz
    ]
    field.setSFVec3f(new_pos)




    pass


    

