import sys
from os import makedirs
from datetime import datetime

# import socnavgym
import gymnasium as gym
import fancy_gym
import numpy as np
import uuid

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common.callbacks import CheckpointCallback

# from transformer_feature_extractor import TransformerFE
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms import bc
from imitation.data import types
from imitation.data import rollout
from imitation.util.util import save_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.evaluation import evaluate_policy


def read_algo():
    return sys.argv[sys.argv.index("-a") + 1]


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
    rng = np.random.default_rng(0)
    note = "Experiment" if "-n" not in sys.argv else read_note()
    algo = "bc" if "-a" not in sys.argv else read_algo()
    env_id = "fancy/Navigation-v0" if "-e" not in sys.argv else read_env()
    load_path = None if "-l" not in sys.argv else load_agent()
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
        model = bc.reconstruct_policy(load_path)
        env = vec_env_fun(env_fns)

        obs = env.reset()
        ret, rets, counter, counter_ = 0, [], 0, 0
        while counter_ < 1000:
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
    else:
        dataset = np.load('dataset_CrowdNavigationStaticVel-v0.npy')
        vec_env = vec_env_fun(env_fns)
        obs_len = int(np.sum(vec_env.observation_space.shape))
        action_len = np.sum(vec_env.action_space.shape)
        dataset_trajs = []
        log_every = 50
        n_epochs = 20
        start_traj_idx, end_traj_idx = 0, 1
        end_of_traj_idxs = np.where(
            np.logical_or(dataset[:, -2] == 1, dataset[:, -1] == 1)
        )[0]
        print("Observation", vec_env.observation_space.shape)
        while end_traj_idx < len(dataset):
            if len(end_of_traj_idxs) == 0:
                assert start_traj_idx + 1 == end_traj_idx
                dataset_trajs.append(types.Trajectory(
                    np.concatenate([
                        dataset[start_traj_idx:, :obs_len],
                        dataset[-1:, obs_len:obs_len * 2]
                    ]),
                    dataset[
                        start_traj_idx:, obs_len * 2:obs_len * 2 + action_len
                    ],
                    None,
                    dataset[-1, -2]
                ))
                break
            elif end_traj_idx == end_of_traj_idxs[0]:
                dataset_trajs.append(types.Trajectory(
                    np.concatenate([
                        dataset[start_traj_idx:end_traj_idx + 1, :obs_len],
                        dataset[end_traj_idx:end_traj_idx + 1, obs_len:obs_len * 2]
                    ]),
                    dataset[
                        start_traj_idx:end_traj_idx + 1,
                        obs_len * 2:obs_len * 2 + action_len
                    ],
                    None,
                    dataset[end_traj_idx, -2]
                ))
                start_traj_idx = end_traj_idx + 1
                end_traj_idx += 1
                end_of_traj_idxs = np.delete(end_of_traj_idxs, 0)
            end_traj_idx += 1
        dataset_trajs = rollout.flatten_trajectories(dataset_trajs)
        print("Dataset length", len(dataset_trajs.obs))

        if algo == "bc":
            bc_trainer = bc.BC(
                observation_space=vec_env.observation_space,
                action_space=vec_env.action_space,
                demonstrations=dataset_trajs,
                rng=rng,
                batch_size=65536,
                optimizer_kwargs={"lr": 5e-4},
            )
            for i in range(log_every):
                bc_trainer.train(n_epochs=n_epochs)
                reward = evaluate_policy(bc_trainer.policy, vec_env, 1000)
                save_policy(
                    bc_trainer.policy,
                    tb_path + "/bc_policy_" + str((i + 1) * n_epochs) + ".pth"
                )
                print("Reward: ", reward)
        elif algo == "gail":
            learner = PPO(
                "MlpPolicy",
                env=vec_env,
                batch_size=64,
                ent_coef=0.0,
                learning_rate=0.0004,
                gamma=0.95,
                n_epochs=5,
                seed=np.random.randint(1000000),
            )
            reward_net = BasicRewardNet(
                observation_space=vec_env.observation_space,
                action_space=vec_env.action_space,
                normalize_input_layer=RunningNorm,
            )
            gail_trainer = GAIL(
                demonstrations=dataset_trajs,
                demo_batch_size=1024 * 16,
                gen_replay_buffer_capacity=512,
                n_disc_updates_per_round=8,
                venv=vec_env,
                gen_algo=learner,
                reward_net=reward_net,
                allow_variable_horizon=True,
            )
            for i in range(log_every):
                gail_trainer.train(10000)
                reward = evaluate_policy(gail_trainer.policy, vec_env, 1000)
                save_policy(
                    gail_trainer.policy,
                    tb_path + "/bc_policy_" + str((i + 1) * n_epochs) + ".pth"
                )
                print("Reward: ", reward)
        else:
            print("Error: algorithm name not recognized!")


if __name__ == '__main__':
    main()
