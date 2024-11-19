import sys
from os import makedirs
from datetime import datetime

import socnavgym
import gymnasium as gym
import fancy_gym
import numpy as np

import stable_baselines3 as sbl
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from transformer_feature_extractor import TransformerFE


def read_algo():
    return sys.argv[sys.argv.index("-a") + 1]


def read_env():
    return sys.argv[sys.argv.index("-e") + 1]


def load_agent():
    return sys.argv[sys.argv.index("-l") + 1]


def read_note():
    return sys.argv[sys.argv.index("-n") + 1]


def tb_time():
    return datetime.now().strftime("%d_%m_%Y-%H:%M:%S")


def tb_custom():
    return sys.argv[sys.argv.index("-p") + 1]


def num_env():
    return sys.argv[sys.argv.index("-ne") + 1]


algo = "ppo" if "-a" not in sys.argv else read_algo()
note = "Experiment" if "-n" not in sys.argv else read_note()
env_id = "fancy_ProDMP/Navigation-v0" if "-e" not in sys.argv else read_env()
load_path = None if "-l" not in sys.argv else load_agent()
tb_path = tb_time() if "-p" not in sys.argv else tb_custom()
tb_path = "exp/" + env_id.replace("fancy/", "") + "/" + algo + "/" + tb_path
test = "-t" in sys.argv
n_envs = 8 if "-ne" not in sys.argv else int(num_env())
np.random.seed()

makedirs(tb_path, exist_ok=True) if not test else None
if not test:
    file_note = open(tb_path + "/note", "a")
    file_note.write(note)
    file_note.close()

if test:
    level = load_path.count("/")
    steps = load_path.split("/")[-1].split("_")[2]
    env_path = "/".join(load_path.split("/")[:level]) +\
        "/rl_model_vecnormalize_" + steps + "_steps.pkl"
    env = VecNormalize.load(env_path, make_vec_env(
        env_id,
        n_envs=1,
        # env_kwargs={"config": "./socnav_env_configs/exp1_no_sngnn.yaml"}
    ))
    env.training = False
    env.norm_reward = False
    model = getattr(sbl, algo.upper()).load(load_path, env=env)

    obs = env.reset()
    ret = 0
    rets = []
    counter = 0
    counter_ = 0
    while counter_ < 23:
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        # input()
        ret += rewards

        if dones:
            # print("Episode return: ", ret)
            rets.append(ret)
            counter += 1 if ret < -10 else 0
            counter_ += 1
            print(counter_)
            ret = 0
            obs = env.reset()
    print("episodes", counter_)
    print("mean", np.mean(rets))
else:
    save_callback = CheckpointCallback(
        5000000 / n_envs, tb_path + "/model_" + algo, save_vecnormalize=True
    )
    if load_path is not None:
        level = load_path.count("/")
        steps = load_path.split("/")[-1].split("_")[2]
        env_path = "/".join(load_path.split("/")[:level]) +\
            "/rl_model_vecnormalize_" + steps + "_steps.pkl"
        vec_env = VecNormalize.load(env_path, make_vec_env(
            env_id,
            n_envs=n_envs,
            # env_kwargs={"config": "./socnav_env_configs/exp1_no_sngnn.yaml"}
        ))
        vec_env.training = True
        vec_env.norm_reward = True
        model = getattr(sbl, algo.upper()).load(load_path, env=vec_env)
    else:
        vec_env = VecNormalize(make_vec_env(
            env_id,
            n_envs=n_envs,
            # env_kwargs={"config": "./socnav_env_configs/exp1_no_sngnn.yaml"}
        ),
            norm_obs=True
        )
        kwargs = {
            "policy_kwargs": {
                "log_std_init": 1.0,
            },
            # max_grad_norm=10.0,
            "clip_range": 0.2,
            "clip_range_vf": 0.2,
            "learning_rate": 1e-4,
            # ent_coef=0.0002,
            # vf_coef=10,
            # target_kl=0.01,
            # gae_lambda=0.97,
            # n_epochs=50,
            "n_steps": 131072 // n_envs,
        } if algo.upper() == "PPO" else {}
        if algo.upper() == "PPO" and "Seq" in env_id:
            kwargs["policy_kwargs"]["features_extractor_class"] = TransformerFE
            kwargs["policy_kwargs"]["features_extractor_kwargs"] = {
                "feature_dim": 16, "input_dim": 2, "n_head": 2, "dim_feedforward": 64
            }
        model = getattr(sbl, algo.upper())(
            "MlpPolicy", vec_env,
            **kwargs,
            verbose=1,
            tensorboard_log=tb_path,
            seed=np.random.randint(1000000),
        )
    model.learn(total_timesteps=50000000, callback=save_callback)
