
from RL_test_env_0722 import World
from controller import Supervisor
from ikpy.chain import Chain
from gym.spaces import Box
import tensorflow as tf

import torch
torch.autograd.set_detect_anomaly(True)
import numpy as np
import datetime

# for training
from stable_baselines3 import PPO   
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import pandas as pd 
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import CallbackList

import os
import matplotlib.pyplot as plt

# -------------------------------
# # 1. 準備 log 目錄與檔名
# log_dir = "logs"
# os.makedirs(log_dir, exist_ok=True)
# monitor_path = os.path.join(log_dir, "ppo_log.csv")

# # 2. 建環境並包 Monitor (+ DummyVecEnv)
# base_env = World()
# env = Monitor(base_env, filename=monitor_path)
# # 如果你習慣向量化，也可以這樣：
# vec_env = DummyVecEnv([lambda: env])

# # -------------------------------
# # 3. 建立並訓練 PPO
# model = PPO(
#     policy='MlpPolicy',
#     env=vec_env,         # 也可直接傳 env
#     device='cpu',
#     learning_rate=3e-4,
#     n_steps=2048,
#     batch_size=64,
#     gamma=0.99,
#     verbose=1
# )
# max_episodes = 50
# for ep in range(max_episodes):
#     model.learn(total_timesteps=env.20)  # 每次跑 N 個 step 當一個 episode
#     # print(f"Episode {ep+1}/{max_episodes} 完成，已訓練 {env.20} 步")
#     print(ep)
# # model.learn(total_timesteps=200)
# model.save('ppo_grinding_feed')

# # -------------------------------
# # 4. 訓練結束後，讀取 log 並繪製 reward 趨勢
# #  4.1 確認檔案存在
# if not os.path.exists(monitor_path):
#     raise FileNotFoundError(f"找不到監控檔：{monitor_path}")

# #  4.2 讀 CSV（跳過 # 開頭的註解行）
# df = pd.read_csv(monitor_path, comment='#')

# #  4.3 繪圖
# plt.figure(figsize=(8,5))
# # x 軸：累積時間步（episode 長度累加）
# plt.plot(df["l"].cumsum(), df["r"], marker='.', linestyle='-')
# plt.xlabel("Time step")
# plt.ylabel("Episode Reward")
# plt.title("訓練結束後的 Reward 趨勢圖")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

###

world = World()

# 2. 建立模型
model = PPO(
    policy='MlpPolicy',
    env=world,
    device='cpu',
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1
)

# 3. 開始訓練
model.learn(total_timesteps=200_000)

# 4. 儲存
model.save('ppo_grinding_feed')







# if torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"
#     print("CUDA is not available, falling back to CPU.")

# class CustomTensorboardCallback(BaseCallback):
#     def __init__(self, log_freq=10, verbose=0):
#         super(CustomTensorboardCallback, self).__init__(verbose)
#         log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ "_no_crash"
#         print("log_dir:", log_dir)
#         # self.writer = SummaryWriter(log_dir="./ppo_tensorboard_log")
#         self.writer = SummaryWriter(log_dir)
#         self.log_freq = log_freq
        

#     def _on_step(self) -> bool:
        
#         timestep = self.num_timesteps

#         # 只有每 log_freq 個 timestep 才寫入
#         if timestep % self.log_freq != 0:
#             return True
        
#         info = self.locals.get("infos", [{}])[0]  # 取第一個環境的 info

#         if "timestep rewards" in info:
#             for i, r in enumerate(info["timestep rewards"]):
#                 self.writer.add_scalar(f"rewards/r{i+1}", r, timestep)

#         if "min errors" in info:
#             for i, e in enumerate(info["min errors"]):
#                 self.writer.add_scalar(f"min errors/error{i+1}", e, timestep)
        
#         if "episode reward" in info:
#             self.writer.add_scalar("episode/episode_reward", info["episode reward"], timestep)

#         if "sample point number" in info:
#             self.writer.add_scalar("episode/sample point number", info["sample point number"], timestep)
#         return True

#     def _on_rollout_end(self) -> None:
#         # 這裡可以計算每一個 episode 的平均 reward
#         ep_rewards = self.locals.get("ep_info_buffer", [])
#         if len(ep_rewards) > 0:
#             mean_ep_reward = sum([ep["r"] for ep in ep_rewards]) / len(ep_rewards)
#             self.writer.add_scalar("episode/mean_reward", mean_ep_reward, self.num_timesteps)

#     def _on_training_end(self) -> None:
#         self.writer.close()

# world = World()


# #若要檢視訓練結果，進入webots環境後，在terminal輸入tensorboard --logdir=儲存數據資料夾的絕對路徑
# #輸入：tensorboard --logdir=./ppo_tensorboard_log/

# # 設定時鐘（time step）
# # timestep = int(world.timestep)


# model = PPO(
#     policy = "MlpPolicy", 
#     env= world, 
#     device="cuda",
#     batch_size=512,
#     learning_rate= 3e-4,  #預設3e-4
#     n_steps= 2048,
    
#     # 增加探索性
#     # ent_coef=0.01              # ✅ 增加熵來鼓勵策略的隨機性
#     # gamma=0.95                 # ✅ 減少折扣因子，重視短期獎勵
#     # gae_lambda=0.9              # ✅ 減少 GAE 平滑，增加估值變動性
#     )

# # 設定訓練參數與 checkpoint callback
# steps = 2048
# episodes = 100000
# total_timesteps = steps * episodes

# checkpoint_callback = CheckpointCallback(
#     save_freq=steps * episodes // 100,  # 每訓練 1% 儲存一次
#     # save_path="./models/",
#     save_path= r"C:\Users\vivian\OneDrive - NTHU\桌面\ITRI\itri_2025project_RL_test\controllers\controller_RL_test",
#     name_prefix="ppo2_train"
# )
# callback = CallbackList([
#     checkpoint_callback,
#     CustomTensorboardCallback()  # 你自己寫的 callback
# ])

# print("Policy on device:", model.policy.device)
# model.learn(total_timesteps= steps*episodes, tb_log_name= "PPO_log2_test", callback=callback)   







