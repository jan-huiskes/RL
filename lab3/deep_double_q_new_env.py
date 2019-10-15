import os
import numpy as np
import matplotlib.pyplot as plt
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

import random
import time
from collections import defaultdict

import gym


def tqdm(*args, **kwargs):
        return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer


## The Q network
class QNetwork(nn.Module):
    def __init__(self, num_hidden=256):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(6, num_hidden) # CartPole 4, car 2
        self.l2 = nn.Linear(num_hidden, 2)


    def forward(self, x):
        # YOUR CODE HERE
        x = self.l1(x)
        x= torch.relu(x)
        x = self.l2(x)

        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []


    def push(self, transition):
        if len(self.memory) >= self.capacity:
            del self.memory[0]
        self.memory.append(transition)


    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        return sample


    def __len__(self):
        return len(self.memory)


# THIS IS VERY IMPORTANT
def get_epsilon(it):
    if it < 40000:
        return 1 - 0.99 * it / 40000
    else:
        return 0.01


def get_epsilon2(it):
    if it < 40000:
        return 1 - 0.99 * it / 40000
    else:
        return 0.01


def select_action(model, state, epsilon):
    with torch.no_grad():
        actions = model(torch.FloatTensor(state))
    actions = actions.numpy()
    return int(np.random.rand() * len(actions)) if np.random.rand() < epsilon else np.argmax(actions)


def compute_q_val(model, state, action):
    Qval = model(torch.FloatTensor(state))
    return torch.gather(Qval, 1, action.unsqueeze(-1)).reshape(-1)


def compute_target(model, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    Qval = model(torch.FloatTensor(next_state))
    target = reward + discount_factor * Qval.max(1)[0] * (1- done.float())

    return target


def train(model, memory, optimizer, batch_size, discount_factor):
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())


def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    optimizer = optim.Adam(model.parameters(), learn_rate)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        episode = 0
        tot_r = 0
        s = env.reset()
        while True:
            eps = get_epsilon(global_steps)
            action = select_action(model, s, eps)

            next_state, reward, done, _ = env.step(action)
            episode += 1
            memory.push((s, action, reward, next_state, done))

            train(model, memory, optimizer, batch_size, discount_factor)

            global_steps += 1

            if done:
                break

            s = next_state
        episode_durations.append(episode)
    return episode_durations


def compute_q_val_double(model, target, state, action):
    Qval = model(torch.FloatTensor(state))
    Qval2 = target(torch.FloatTensor(state))
    return torch.gather(Qval, 1, action.unsqueeze(-1)).reshape(-1), torch.gather(Qval2, 1, action.unsqueeze(-1)).reshape(-1)


def compute_target_double(model, target, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    next_Q1 = model(torch.FloatTensor(next_state))
    next_Q2 = target(torch.FloatTensor(next_state))
    next_Q = torch.min(
        torch.max(next_Q1, 1)[0],
        torch.max(next_Q2, 1)[0]
    )
    next_Q = next_Q.view(next_Q.size(0), 1)
    expected_Q = reward + discount_factor * next_Q.max(1)[0] * (1- done.float())

    return expected_Q


def train_double(model, target, memory, optimizer,optimizer2, batch_size, discount_factor):
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean

    # compute the q value
    q_val, q_val2 = compute_q_val_double(model, target, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        expected_Q = compute_target_double(model, target, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    loss1 = F.mse_loss(q_val, expected_Q)

    loss2 = F.mse_loss(q_val2, expected_Q)
    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()

    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()
    return loss1.item()  # Returns a Python scalar, and releases history (similar to .detach())


def run_episodes_double(train, model,target, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    optimizer1 = optim.Adam(model.parameters(), learn_rate)
    optimizer2 = optim.Adam(target.parameters(), learn_rate)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        episode = 0
        s = env.reset()
        while True:
            eps = get_epsilon2(global_steps)
            action = select_action(model, s, eps)

            next_state, reward, done, _ = env.step(action)
            # TODO: is it correct to compute duration here and not after the break?
            episode += 1

            memory.push((s, action, reward, next_state, done))

            train_double(model, target, memory, optimizer1, optimizer2, batch_size, discount_factor)

            global_steps += 1

            if done:
                break

            s = next_state
        episode_durations.append(episode)
    return episode_durations


def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def set_seeds(env, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    np.random.seed(seed)


def run_single(seeds, params):
    results = []
    memory, env, num_episodes, batch_size, discount_factor, learn_rate, num_hidden = params

    for i, seed in enumerate(seeds):
        print("Doing run %i" % i)
        set_seeds(env, seed)
        model = QNetwork(num_hidden)
        episode_durations = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)
        results.append(episode_durations)

    return results


def run_double(seeds, params):
    results = []
    memory, env, num_episodes, batch_size, discount_factor, learn_rate, num_hidden = params

    for i, seed in enumerate(seeds):
        print("Doing run %i" % i)
        set_seeds(env, seed)
        model = QNetwork(num_hidden)
        target = QNetwork(num_hidden)
        episode_durations_double = run_episodes_double(train, model, target, memory, env, num_episodes, batch_size, discount_factor, learn_rate)
        results.append(episode_durations_double)

    return results


def gen_seeds(seed, nr_runs):
    seeds = []
    random.seed(seed)

    for i in range(nr_runs):
        seeds.append(random.randint(0, 1000))
    return seeds


def error(data):
    avg = np.average(data, 0)
    std = np.std(data, 0)
    return avg, std


if __name__ == '__main__':
    # Params.
    memory = ReplayMemory(200000)
    env = gym.envs.make("Acrobot-v1") #CartPole-v0
    num_episodes = 200
    batch_size = 256
    discount_factor = 0.99
    learn_rate = 0.0001
    num_hidden = 128

    params = (memory, env, num_episodes, batch_size, discount_factor, learn_rate, num_hidden)

    # Seed management
    nr_runs = 10
    seed = 42
    seeds = gen_seeds(seed, nr_runs)

    print('Running Single Q')
    returns_single = run_single(seeds, params)

    print('Running Double Q')
    returns_double = run_double(seeds, params)

    print('Plotting')
    returns_single = np.asarray(returns_single)
    avg_single, std_single = error(returns_single)

    returns_double = np.asarray(returns_double)
    avg_double, std_double = error(returns_double)


    plt.plot(avg_single, color='b', label='DQN')
    plt.fill_between(list(range(len(avg_single))), avg_single-std_single, avg_single+std_single, alpha=0.5)
    plt.plot(avg_double, color='r', label='DDQN')
    plt.fill_between(list(range(len(avg_double))), avg_double-std_double, avg_double+std_double, alpha=0.5)
    plt.legend()
    plt.xlabel("Episode (#)")
    plt.ylabel("Episode Length")
    plt.title("Deep Single Q vs Deep Double Q on the Acrobot problem")
    plt.show()

    # for episode_durations in returns_single:
    #     plt.plot(smooth(episode_durations, 10),color='b', label='DQN')
    # for episode_durations_double in returns_double:
    #     plt.plot(smooth(episode_durations_double, 10),color='r', label='Double DQN')
    # plt.legend()
    # plt.xlabel("Episode (#)")
    # plt.ylabel("Episode Length")
    # plt.title("Deep Single Q vs Deep Double Q on the CartPole problem")
    # plt.show()
