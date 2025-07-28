# from ddpg_pendulum import *
from controller_RL_testttt_env import World
# from ddpg import DDPG as Agent
from controller import Robot, Supervisor,Connector
from ikpy.chain import Chain
from gym.spaces import Box
import tensorflow as tf

import torch
torch.autograd.set_detect_anomaly(True)
import numpy as np
import torch
import datetime

# for training
from stable_baselines3 import PPO   
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import pandas as pd 
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import CallbackList

import os
import datetime


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("CUDA is not available, falling back to CPU.")

class CustomTensorboardCallback(BaseCallback):
    def __init__(self, log_freq=10, verbose=0):
        super(CustomTensorboardCallback, self).__init__(verbose)
        log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ "_no_crash"
        print("log_dir:", log_dir)
        # self.writer = SummaryWriter(log_dir="./ppo_tensorboard_log")
        self.writer = SummaryWriter(log_dir)
        self.log_freq = log_freq
        

    def _on_step(self) -> bool:
        
        timestep = self.num_timesteps

        # 只有每 log_freq 個 timestep 才寫入
        if timestep % self.log_freq != 0:
            return True
        
        info = self.locals.get("infos", [{}])[0]  # 取第一個環境的 info

        if "timestep rewards" in info:
            for i, r in enumerate(info["timestep rewards"]):
                self.writer.add_scalar(f"rewards/r{i+1}", r, timestep)

        if "min errors" in info:
            for i, e in enumerate(info["min errors"]):
                self.writer.add_scalar(f"min errors/error{i+1}", e, timestep)
        
        if "episode reward" in info:
            self.writer.add_scalar("episode/episode_reward", info["episode reward"], timestep)

        if "sample point number" in info:
            self.writer.add_scalar("episode/sample point number", info["sample point number"], timestep)
        return True

    def _on_rollout_end(self) -> None:
        # 這裡可以計算每一個 episode 的平均 reward
        ep_rewards = self.locals.get("ep_info_buffer", [])
        if len(ep_rewards) > 0:
            mean_ep_reward = sum([ep["r"] for ep in ep_rewards]) / len(ep_rewards)
            self.writer.add_scalar("episode/mean_reward", mean_ep_reward, self.num_timesteps)

    def _on_training_end(self) -> None:
        self.writer.close()

world = World()


#若要檢視訓練結果，進入webots環境後，在terminal輸入tensorboard --logdir=儲存數據資料夾的絕對路徑
#輸入：tensorboard --logdir=./ppo_tensorboard_log/

# 設定時鐘（time step）
timestep = int(world.timestep)

model = PPO(
    policy = "MlpPolicy", 
    env= world, 
    device="cuda",
    batch_size=512,
    learning_rate= 3e-4,  #預設3e-4
    n_steps= 2048,
    
    # 增加探索性
    # ent_coef=0.01              # ✅ 增加熵來鼓勵策略的隨機性
    # gamma=0.95                 # ✅ 減少折扣因子，重視短期獎勵
    # gae_lambda=0.9              # ✅ 減少 GAE 平滑，增加估值變動性
    )

# 設定訓練參數與 checkpoint callback
steps = 2048
episodes = 100000
total_timesteps = steps * episodes

checkpoint_callback = CheckpointCallback(
    save_freq=steps * episodes // 100,  # 每訓練 1% 儲存一次
    # save_path="./models/",
    save_path= r"C:\Users\vivian\OneDrive - NTHU\桌面\ITRI\webot_practice_2025ver\controllers\controller_RL_test",
    name_prefix="ppo2_train"
)
callback = CallbackList([
    checkpoint_callback,
    CustomTensorboardCallback()  # 你自己寫的 callback
])

print("Policy on device:", model.policy.device)
model.learn(total_timesteps= steps*episodes, tb_log_name= "PPO_log2_test", callback=callback)   

################################################################################################################################

# # 載入之前訓練好的模型

# env = DummyVecEnv([lambda: world])          # SB3 需要 VecEnv 格式

# # 載入已訓練的 PPO 模型
# model_path = "models/ppo2_train_6144000_steps"  # <-- 替換成你自己的模型路徑
# model = PPO.load(model_path, env=env, device="cuda")  # 如果有用 GPU 的話

# # 重設環境
# obs = env.reset()

# # 開始執行測試 Episode
# done = False
# episode_reward = 0
# step_count = 0

# while not done:
#     action, _ = model.predict(obs, deterministic=True)  # ✅ 採用最優策略
#     obs, reward, done, info = env.step(action)
#     episode_reward += reward[0]  # reward 是 list（VecEnv 包裝）
#     step_count += 1


# print(f"✅ 測試完成，共 {step_count} 步，總回報為：{episode_reward:.2f}")





