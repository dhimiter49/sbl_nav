import sys
import csv
from pathlib import Path
from os import makedirs
from datetime import datetime

# import socnavgym
import gymnasium as gym
import fancy_gym
import numpy as np
import uuid

import stable_baselines3 as sbl
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from transformer_feature_extractor import TransformerFE
from cnn1d_feature_extractor import CNN1dFE


def read_algo():
    return sys.argv[sys.argv.index("-a") + 1]


def read_exp_name():
    return sys.argv[sys.argv.index("-name") + 1]

def read_env():
    return sys.argv[sys.argv.index("-e") + 1]


def load_agent():
    return sys.argv[sys.argv.index("-l") + 1]


def read_note():
    return sys.argv[sys.argv.index("-n") + 1]


def tb_time():
    return datetime.now().strftime("%d_%m_%Y-%H:%M:%S") + "-" + str(uuid.uuid4().int)[-4:]


def tb_custom():
    return sys.argv[sys.argv.index("-p") + 1]


def num_env():
    return sys.argv[sys.argv.index("-ne") + 1]


def architecture():
    return sys.argv[sys.argv.index("-ar") + 1]


def make_env(env_id: str, **kwargs) -> callable:
    """
    returns callable to create gym environment or monitor

    Args:
        env_id: gym env ID

    Returns: callable for env constructor
    """
    def _get_env():
        env = gym.make(env_id)

        return Monitor(env)

    return _get_env


def main():
    algo = "ppo" if "-a" not in sys.argv else read_algo()
    note = "Experiment" if "-n" not in sys.argv else read_note()
    env_id = "fancy_ProDMP/Navigation-v0" if "-e" not in sys.argv else read_env()
    load_path = None if "-l" not in sys.argv else load_agent()
    arch = "MLP" if "-ar" not in sys.argv else architecture()
    exp_name = algo if "-name" not in sys.argv else read_exp_name()
    tb_path = tb_time() if "-p" not in sys.argv else tb_custom()
    tb_path = "exp/" + env_id.replace("fancy/", "").replace("fancy_ProDMP/", "") +\
        "/" + algo + "/" + tb_path
    test = "-t" in sys.argv
    render = "-r" in sys.argv
    n_envs = 8 if "-ne" not in sys.argv else int(num_env())
    n_envs = 1 if test else n_envs
    np.random.seed()
    vec_env_fun = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env_fns = [make_env(env_id) for i in range(n_envs)]

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
        env = VecNormalize.load(env_path, vec_env_fun(env_fns))
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
            env.render() if render else None
            # input()
            ret += rewards

            if dones:
                # print("Episode return: ", ret)
                rets.append(ret)
                counter += 1 if ret < -10 else 0
                counter_ += 1
                ret = 0
                obs = env.reset()
        print("episodes", counter_)
        print("mean", np.mean(rets))
        print("Stats:")
        (
            col_rate,
            col_speed,
            col_agent_speed,
            avg_intersect_area,
            avg_intersect_area_percent,
            freezing_instances,
            avg_ttg,
            success_rate
        ) = env.env_method("stats")[0]  # only one environment in testing
        exp_name = exp_name + ".csv"
        path = Path.home() / "Documents" / "RAM" / "results" / exp_name
        has_header = False
        if path.is_file():
            with open(path, 'r', newline='') as csvfile:
                sniffer = csv.Sniffer()
                has_header = sniffer.has_header(csvfile.read(2048))
        with open(path, 'a', newline='') as csvfile:
            fieldnames = [
                'return', 'ttg', 'success_rate',
                'col_rate', 'col_speed', 'col_agent_speed',
                'col_intersection_area', 'col_intersection_percent',
                'freezing_instances'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not has_header:
                writer.writeheader()
            writer.writerow({
                "return": np.mean(rets),
                "ttg": avg_ttg,
                "success_rate": success_rate,
                "col_rate": col_rate,
                "col_speed": col_speed,
                "col_agent_speed": col_agent_speed,
                "col_intersection_area": avg_intersect_area,
                "col_intersection_percent": avg_intersect_area_percent,
                "freezing_instances": freezing_instances,
            })
    else:
        save_callback = CheckpointCallback(
            5000000 / n_envs, tb_path + "/model_" + algo, save_vecnormalize=True
        )
        if load_path is not None:
            level = load_path.count("/")
            steps = load_path.split("/")[-1].split("_")[2]
            env_path = "/".join(load_path.split("/")[:level]) +\
                "/rl_model_vecnormalize_" + steps + "_steps.pkl"
            vec_env = VecNormalize.load(env_path, vec_env_fun(env_fns))
            vec_env.training = True
            vec_env.norm_reward = True
            model = getattr(sbl, algo.upper()).load(load_path, env=vec_env)
        else:
            vec_env = VecNormalize(vec_env_fun(env_fns), norm_obs=True)
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
            } if algo.upper() == "PPO" else {
                "policy_kwargs": {}
            }
            if "Seq" in env_id:
                kwargs["policy_kwargs"]["features_extractor_class"] = TransformerFE
                kwargs["policy_kwargs"]["features_extractor_kwargs"] = {
                    "feature_dim": 16, "input_dim": 2, "n_head": 2, "dim_feedforward": 64
                }
            if arch == "1dcnn":
                kwargs["policy_kwargs"]["features_extractor_class"] = CNN1dFE
                # kwargs["policy_kwargs"]["features_extractor_kwargs"] = {
                #     "channels": 4, "kernel_size": 5, "non_lidar_dim": 4, "one_cnn": True
                # }
            model = getattr(sbl, algo.upper())(
                "MlpPolicy", vec_env,
                **kwargs,
                verbose=1,
                tensorboard_log=tb_path,
                seed=np.random.randint(1000000),
            )
        model.learn(total_timesteps=50000000, callback=save_callback)


if __name__ == '__main__':
    main()
