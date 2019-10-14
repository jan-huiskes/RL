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

from windy_gridworld import WindyGridworldEnv


"""
For my own sanity
Have replaced "Q = defaultdict" with "Q = keydefaultdict" as a hack to create different
size zero arrays based on how many actions exist for any specific state.

I have then in "make_epsilon_greedy_policy()" overridden the passed parameter nA to also
take into account the number of actions available in each state.

I have botched the "Actions.step()" function.

I have moved the "if done: break" statement in the learn functions so that we do still
learn from the terminal step (which we didn't before, which is probably bad?). To do
this I had to fake there being 1 action from the terminal state, so that the Q function
has a value for the terminal state. That value being zero.
"""


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret
        

# Defines number of actions. Note that we cannot pass a state here.
class Actions:
    def __init__(self):
        self.actions = [2, 99, 1]

    def n(self, state):
        return self.actions[state]


# Defines the environment.
class Bats:
    def __init__(self):
        self.action_space = Actions()
        self.state = 0

    def reset(self):
        self.state = 0
        return 0

    def step(self, action):
        if self.state == 0:
            if action == 0:
                self.state = 2
                return 2, -10, True, None
            elif action == 1:
                self.state = 1
                return 1, 0, False, None
            else:
                print('what?')
        elif self.state == 1:
            self.state = 2
            return 2, random.choice([-5, -30]), True, None
        else:
            print('what??')

    # Does fuck all.
    def seed(self, seed):
        pass


def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    """
    def policy_fn(observation):
        nA = len(Q[observation])
        return int(np.random.rand() * nA) if np.random.rand() < epsilon else np.argmax(Q[observation])
    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, Q=None):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy


    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Probability to sample a random action. Float between 0 and 1.
        Q: hot-start the algorithm with a Q value function (optional)

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is a list of tuples giving the episode lengths and rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    if Q is None:
        Q = keydefaultdict(lambda x: np.zeros(env.action_space.n(x)))

    # Keeps track of useful statistics
    stats = []

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    pol_greed = make_epsilon_greedy_policy(Q, 0, env.action_space.n)

    for i_episode in range(num_episodes):
        i = 0
        R = 0

        s = env.reset()
        while True:
            a = policy(s)
            s_next, reward, done, _ = env.step(a)
            i += 1
            R += reward

            # FIXME if our whole episode is just one step, we don't learn.
            # if done:
            #    break

            a_next = pol_greed(s_next)

            Q[s][a] = alpha * (reward + discount_factor * Q[s_next][a_next] - Q[s][a]) + Q[s][a]
            s = s_next

            # FIXME so i moved this here.
            if done:
                break

        stats.append((i, R))

    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)


def double_q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, Q=None):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy


    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Probability to sample a random action. Float between 0 and 1.
        Q: hot-start the algorithm with a Q value function (optional)

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is a list of tuples giving the episode lengths and rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    # Q1, Q2, and one combined for eps greedy policy
    if Q is None:
        Q1 = keydefaultdict(lambda x: np.zeros(env.action_space.n(x)))
        Q2 = keydefaultdict(lambda x: np.zeros(env.action_space.n(x)))
        combined_Q = keydefaultdict(lambda x: np.zeros(env.action_space.n(x)))

    # Keeps track of useful statistics
    stats = []

    # The policy we're following based on combined Q values
    policy = make_epsilon_greedy_policy(combined_Q, epsilon, env.action_space.n)

    # Different greedy policies
    pol_greed_1 = make_epsilon_greedy_policy(Q1, 0, env.action_space.n)
    pol_greed_2 = make_epsilon_greedy_policy(Q2, 0, env.action_space.n)

    for i_episode in range(num_episodes):
        i = 0
        R = 0

        s = env.reset()
        while True:
            a = policy(s)
            s_next, reward, done, _ = env.step(a)
            i += 1
            R += reward

            # FIXME if episode is one long, we don't learn.
            #if done:
            #    break

            # Random pick one, update and update the combined Dictionary
            if np.random.uniform() < 0.5:
                a_next = pol_greed_2(s_next)
                Q2[s][a] = alpha * (reward + discount_factor * Q1[s_next][a_next] - Q2[s][a]) + Q2[s][a]
                combined_Q[s][a] = alpha * (reward + discount_factor * combined_Q[s_next][a_next] - combined_Q[s][a]) + combined_Q[s][a]
                s = s_next
            else:
                a_next = pol_greed_1(s_next)
                Q1[s][a] = alpha * (reward + discount_factor * Q2[s_next][a_next] - Q1[s][a]) + Q1[s][a]
                combined_Q[s][a] = alpha * (reward + discount_factor * combined_Q[s_next][a_next] - combined_Q[s][a]) + combined_Q[s][a]
                s = s_next

            # FIXME so i moved this here.
            if done:
                break

        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q1, (episode_lengths, episode_returns)


def set_seeds(env, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    np.random.seed(seed)


def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def gen_seeds(seed, nr_runs):
    seeds = []
    random.seed(seed)

    for i in range(nr_runs):
        seeds.append(random.randint(0, 1000))

    return seeds


def run_single(seeds, params):
    results = []
    env = params

    for seed in seeds:
        set_seeds(env, seed)
        Q_q_learning, (episode_lengths_q_learning, episode_returns_q_learning) = q_learning(env, 300)
        results.append(episode_returns_q_learning)

    return results


def run_double(seeds, params):
    results = []
    env = params

    for seed in seeds:
        set_seeds(env, seed)
        Q_double_q_learning, (episode_lengths_double_q_learning, episode_returns_double_q_learning) = double_q_learning(env, 300)
        results.append(episode_returns_double_q_learning)

    return results


def error(data):
    avg = np.average(data, 0)
    std = np.std(data, 0)
    return avg, std


if __name__ == '__main__':
    #env = WindyGridworldEnv()
    env = Bats()
    params = env

    nr_runs = 100
    seed = 42
    seeds = gen_seeds(seed, nr_runs)
    
    returns_single = run_single(seeds, params)
    returns_double = run_double(seeds, params)

    returns_single = np.asarray(returns_single)
    avg_single, std_single = error(returns_single)

    returns_double = np.asarray(returns_double)
    avg_double, std_double = error(returns_double)

    plt.plot(avg_single, color='b', label='Single Q')
    plt.fill_between(list(range(len(avg_single))), avg_single-std_single, avg_single+std_single, alpha=0.5)
    plt.plot(avg_double, color='r', label='Double Q')
    plt.fill_between(list(range(len(avg_double))), avg_double-std_double, avg_double+std_double, alpha=0.5)
    plt.legend()
    plt.xlabel("Episode (#)")
    plt.ylabel("Time penalty (min)")
    plt.title("Single Q vs Double Q on the mountain bat problem")
    plt.show()

    #for return_single in returns_single: 
    #    plt.plot(smooth(return_single, 10),color='b', label='Q')
    #for return_double in returns_double: 
    #    plt.plot(smooth(return_double, 10),color='r', label='DQ')
    #plt.legend()
    #plt.title('smoothed rewards')
    #plt.show()

    #plt.plot(returns_single[0])
    #plt.ylabel('return (travel time)')
    #plt.xlabel('episodes')
    #plt.title('Episode returns Q-learning')
    #plt.show()

    #plt.plot(returns_double[0])
    #plt.ylabel('return (travel time)')
    #plt.xlabel('episodes')
    #plt.title('Episode returns double Q-learning')
    #plt.show()


