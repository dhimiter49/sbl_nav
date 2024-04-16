import sys
from os import mkdir
from datetime import datetime

import gymnasium as gym
import fancy_gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
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
tb_path = "exp/" + tb_path + "_MP"
test = "-t" in sys.argv
n_envs = 8 if "-ne" not in sys.argv else int(num_env())

mkdir(tb_path)
save_callback = CheckpointCallback(
    500000 / n_envs, tb_path + "/model_ppo", save_vecnormalize=True
)

if test:
    steps = load_path.split("/")[-1].split("_")[2]
    env_path = "/".join(load_path.split("/")[:3]) +\
        "/rl_model_vecnormalize_" + steps + "_steps.pkl"
    env = VecNormalize.load(env_path, make_vec_env(env_id, n_envs=1))
    env.training = False
    env.norm_reward = False
    model = PPO.load(load_path, env=env)

    obs = env.reset()
    ret = 0
    for i in range(10000):
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        ret += rewards

        if dones:
            print("Episode return: ", ret)
            ret = 0
            obs = env.reset()
else:
    if load_path is not None:
        steps = load_path.split("/")[-1].split("_")[2]
        env_path = "/".join(load_path.split("/")[:3]) +\
            "/rl_model_vecnormalize_" + steps + "_steps.pkl"
        vec_env = VecNormalize.load(env_path, make_vec_env(env_id, n_envs=n_envs))
        env.training = True
        env.norm_reward = True
        model = PPO.load(load_path, env=vec_env)
    else:
        vec_env = VecNormalize(make_vec_env(env_id, n_envs=n_envs), norm_obs=True)
        model = PPO(
            "MlpPolicy", vec_env,
            policy_kwargs={
                "net_arch": {"pi": [64, 64], "vf": [128, 128]},
                "log_std_init": 2.0,
            },
            max_grad_norm=10.0,
            clip_range=0.2,
            clip_range_vf=0.2,
            learning_rate=1e-4,
            verbose=1,
            n_steps=1024 // n_envs,
            tensorboard_log=tb_path
        )
    model.learn(total_timesteps=10000000, callback=save_callback)
