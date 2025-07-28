# """
# force_check.py ─ 自動驗證單軸 TouchSensor 線性度
# ‧ Webots 2025 ‧ basicTimeStep=8 ms
# """
# from controller import Robot, Supervisor
# import numpy as np

# TIME_STEP = 8                       # 必須等於 basicTimeStep
# # MASSES   = [0.0, 0.5, 1.0, 1.5, 2.0]  # kg，0 代表零點
# MASSES   = [ 1.0 ]  # kg，0 代表零點
# Z_DROP   = 0.11                     # 砝碼瞬移高度 (m)
# SETTLE_STEPS = 250                  # 8 ms × 250 ≒ 2 s

# bot  = Supervisor()                 # 若不是 supervisor，可直接用 Robot()
# fs   = bot.getDevice('load_cell')
# fs.enable(TIME_STEP)

# # 取得砝碼節點（預先在 .wbt DEF 好）
# def_node = {m: bot.getFromDef(f"Mass{m}kg") for m in MASSES if m!=0}

# data = []                           # 儲存 (mass, force)
# # ------------------------------------------------------------------- #
# # 2. 逐一測試質量
# for m in MASSES:

#     # 2-1 把上一次砝碼移開
#     for mass, node in def_node.items():
#         node.getField('translation').setSFVec3f([0, 0, 0.40])  # 空中停放

#     # 2-2 如果 m>0，將對應砝碼「瞬移」到感測面上方
#     if m > 0:
#         def_node[m].getField('translation').setSFVec3f([0, 0, Z_DROP])

#     # 3. 等待系統振動衰減
#     for _ in range(SETTLE_STEPS):
#         bot.step(TIME_STEP)

#     # 4. 連續讀 20 筆取平均
#     readings = []
#     for _ in range(20):
#         bot.step(TIME_STEP)
#         readings.append(fs.getValue())
#     f_mean = sum(readings) / len(readings)

#     print(f"質量 {m:3.1f} kg ➜ 感測值 {f_mean:6.2f} N")
#     data.append((m, f_mean))

# # ------------------------------------------------------------------- #
# # 5. 線性回歸  F ≈ k·m + b
# m_arr, f_arr = np.array(data).T
# k, b = np.polyfit(m_arr, f_arr, 1)

# # 6. 列印報告
# print("\n------ 測試報告 ------")
# print(f"斜率 k = {k:5.3f} N/kg   (理論 9.81)")
# print(f"截距 b = {b:5.3f} N")
# for m, f_meas in data:
#     f_theo = m * 9.81
#     err = 0 if m==0 else 100*(f_meas - f_theo)/f_theo
#     print(f"{m:3.1f} kg : 量測 {f_meas:6.2f} N / 理論 {f_theo:6.2f} N → 誤差 {err:5.2f} %")


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

import warnings

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())  # 取得基本時間步長

# 取得力感測器
force_sensor = supervisor.getDevice("force sensor")
if force_sensor is None:
    print(" ERROR: 'force_sensor' not found!")
    exit(1)
    
force_sensor.enable(timestep)  # 啟用感測器

# 準備儲存清單
data = []    # 每筆： [time(s), fx, fy, fz]
t = 0.0

while supervisor.step(timestep) != -1:
    # 三軸
    force_values = force_sensor.getValues()  # 取得感測數據
    # print(f"End-Effector Force: X={force_values[0]:.3f}, Y={force_values[1]:.3f}, Z={force_values[2]:.3f}")
    fx, fy, fz = force_sensor.getValues()[:3]         # 讀取三軸力
    print("fx={:.3f},fy={:.3f},fz={:.3f}".format(fx, fy, fz))
    data.append([t, fx, fy, fz])

    # 單軸
    # fz = force_sensor.getValue()  # 取得感測數據
    # data.append([t, fz])
    # print(f'Fz = {fz:.3f} N')

    t += timestep / 1000.0  # 更新時間 (毫秒)


# 結束時寫成 CSV
with open('force_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time','fx','fy','fz'])
    writer.writerows(data)
print("已將感測資料儲存到 force_data.csv")

# with open('force_data.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['time','fz'])
#     writer.writerows(data)
# print("已將感測資料儲存到 force_data.csv")
