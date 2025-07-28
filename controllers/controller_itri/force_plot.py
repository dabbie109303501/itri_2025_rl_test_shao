import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

# 讀資料
df = pd.read_csv(r"C:\Users\vivian\OneDrive - NTHU\桌面\ITRI\webot_practice_2025ver\controllers\controller_test_shao\force_data.csv")

# 畫 X、Y、Z 三條曲線

f = np.sqrt(df['fx']**2 + df['fy']**2 + df['fz']**2)
plt.plot(df['time'], f, label='F')

plt.plot(df['time'], df['fx'], label='Fx')
plt.plot(df['time'], df['fy'], label='Fy')
plt.plot(df['time'], df['fz'], label='Fz')

plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force Sensor Readings Over Time')
plt.legend()
plt.grid(True)
plt.show()
