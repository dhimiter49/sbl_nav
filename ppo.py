import sys
from datetime import datetime

import gymnasium as gym
import fancy_gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback


def read_env():
    return sys.argv[sys.argv.index("-e") + 1]

def load_agent():
    return sys.argv[sys.argv.index("-l") + 1]

def tb_time():
    return datetime.now().strftime("%d_%m_%Y-%H:%M:%S")

def tb_custom():
    return sys.argv[sys.argv.index("-p") + 1]

def num_env():
    return sys.argv[sys.argv.index("-ne") + 1]

env_id = "fancy_ProDMP/Navigation-v0" if "-e" not in sys.argv else read_env()
load_path = None if "-l" not in sys.argv else load_agent()
tb_path = tb_time() if "-p" not in sys.argv else tb_custom()
tb_path = "exp/" + tb_path
test = "-t" in sys.argv
n_envs = 8 if "-ne" not in sys.argv else int(num_env())

save_callback = CheckpointCallback(1000000 / n_envs, tb_path + "/model_ppo")

if test:
    model = PPO.load(load_path)

    env = gym.make(env_id)
    obs, _ = env.reset()
    for i in range(10000):
        action, _ = model.predict(obs)
        obs, rewards, dones, terminal, info = env.step(action)
        env.render()

        if dones or terminal:
            obs, _ = env.reset()
else:
    vec_env = make_vec_env(env_id, n_envs=n_envs)
    if load_path is not None:
        model = PPO.load(load_path, env=vec_env)
    else:
        model = PPO(
            "MlpPolicy", vec_env,
            policy_kwargs={
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                "log_std_init": 1.,
                "ortho_init": False
            },
            clip_range=0.2,
            clip_range_vf=0.2,
            learning_rate=3e-4,
            verbose=1,
            tensorboard_log=tb_path
        )

    model.learn(total_timesteps=10000000, callback=save_callback)
