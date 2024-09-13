import sys
import gymnasium as gym
import fancy_gym
from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import stable_baselines3 as sbl
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.data.dataset import Dataset, random_split
from stable_baselines3 import PPO, A2C


class ExpertDataSet(Dataset):
    def __init__(self, dataset, obs_space, action_space):
        self.observations = dataset[:, :obs_space]
        self.actions = dataset[:, obs_space * 2: obs_space * 2 + action_space]


    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])


    def __len__(self):
        return len(self.observations)


def read_algo():
    return sys.argv[sys.argv.index("-a") + 1]


def read_env():
    return sys.argv[sys.argv.index("-e") + 1]


def load_agent():
    return sys.argv[sys.argv.index("-l") + 1]


env_id = "fancy_ProDMP/Navigation-v0" if "-e" not in sys.argv else read_env()
algo = "ppo" if "-a" not in sys.argv else read_algo()
env = gym.make(env_id)
load_path = None if "-l" not in sys.argv else load_agent()
dataset = np.load('dataset.npy')
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
dataset = th.from_numpy(dataset)


dataset = ExpertDataSet(
    dataset, np.sum(env.observation_space.shape), np.sum(env.action_space.shape)
)
train_expert_dataset, test_expert_dataset = random_split(
    dataset, [train_size, test_size]
)

if load_path is not None:
    student = getattr(sbl, algo.upper()).load(load_path, env)
else:
    student = getattr(sbl, algo.upper())(
        "MlpPolicy", env, seed=np.random.randint(1000000)
    )


def pretrain_agent(
    mdoel,
    batch_size=64,
    epochs=1000,
    scheduler_gamma=0.7,
    learning_rate=1.0,
    log_interval=100,
    no_cuda=True,
    seed=1,
    test_batch_size=64,
):
    use_cuda = not no_cuda and th.cuda.is_available()
    th.manual_seed(seed)
    device = th.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    if isinstance(env.action_space, gym.spaces.Box):
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    # Extract initial policy
    model = student.policy.to(device)


    def train(model, device, train_loader, optimizer):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            if isinstance(env.action_space, gym.spaces.Box):
                if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                else:
                    action = model(data)
                action_prediction = action.double()
            else:
                dist = model.get_distribution(data)
                action_prediction = dist.distribution.logits
                target = target.long()
            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()


    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if isinstance(env.action_space, gym.spaces.Box):
                    if isinstance(student, (A2C, PPO)):
                        action, _, _ = model(data)
                    else:
                        # SAC/TD3:
                        action = model(data)
                    action_prediction = action.double()
                else:
                    dist = model.get_distribution(data)
                    action_prediction = dist.distribution.logits
                    target = target.long()
                    test_loss = criterion(action_prediction, target)

        test_loss /= len(test_loader.dataset)

    train_loader = th.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = th.utils.data.DataLoader(
        dataset=test_expert_dataset, batch_size=test_batch_size, shuffle=True, **kwargs
    )
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()
    student.policy = model


mean_reward, std_reward = evaluate_policy(student, env, n_eval_episodes=10)
print(f"Mean reward = {mean_reward} +/- {std_reward}")

pretrain_agent(
    student,
    epochs=1000,
    scheduler_gamma=0.7,
    learning_rate=0.001,
    log_interval=100,
    no_cuda=True,
    seed=1,
    batch_size=64,
    test_batch_size=1000
)

mean_reward, std_reward = evaluate_policy(
    student, env, n_eval_episodes=10
)
student.save("student")
print(f"Mean reward = {mean_reward} +/- {std_reward}")
