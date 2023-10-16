import argparse
from collections import deque
import itertools
import random
import queue

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=32):
        super().__init__()
        ## TODO ##
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ## TODO ##
        x = torch.tensor(x, device="cuda:0")
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        ## TODO ##
        self._optimizer = optim.Adam(self._behavior_net.parameters(), lr=args.lr)

        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
         ## TODO ##
        # selects a random action with probability ε.
        if random.random() < epsilon:
            return action_space.sample()
        else:
            values = self._behavior_net(state)
            # torch.max(input, dim, keepdim = False) Returns (values, indices) : (max, max_indices)
            # values : the maximum value of each row of the input tensor in the given dimension dim. 
            # indices : the index location of each maximum value found (argmax).
            # dim = 0
            _, action = torch.max(values, 0)
            return action.item()
        
            # with torch.no_grad():
            #     action = torch.argmax(self._behavior_net(torch.from_numpy(state).view(1,-1).to(self.device)), dim=1).item()
            # return action

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward / 10], next_state, [int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## TODO ##
        # torch.gather (input, dim, index)
        # dim (int) – the axis along which to index 
        # index (LongTensor) – the indices of elements to gather
        # dim = 1, index = action.long()
        # get the long tensor (value) of dimension 1
        q_value = self._behavior_net(state).gather(1, action.long())

        with torch.no_grad():
           # get the maximum value of dimension 1
           q_next = self._target_net(next_state).detach().max(1)[0].unsqueeze(1)
           # q_next = torch.max(self._target_net(next_state), dim = 1)[0].view(-1,1)
           # q_next = torch.max(self._target_net(next_state), 1)[0].view(-1, 1)
           # (for non-terminal φj+1 => done = 0)  yj = rj + γ maxa′ Q(φj+1, a′; θ) 
           # (for terminal φj+1 => done = 1)      yj = rj 
           q_target = reward + gamma * q_next * (1 - done)

        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)
        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，解決梯度爆炸的問題
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        tau = 0.001
        for target_param, behavior_param in zip(self._target_net.parameters(), self._behavior_net.parameters()):
            target_param.data.copy_(tau * behavior_param.data + (1.0 - tau) * target_param.data)
        
        # updates every 1000 steps
        # self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, env, agent, writer):
    print('Start Training')
    # Action [4]: 0 (No-op), 1 (Fire left engine), 2 (Fire main engine), 3 (Fire right engine)
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0

    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()

        for t in itertools.count(start=1):
            # first 10000 steps : random action
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                # ε annealed linearly from 1 to 0.01 over the first xxx frames, and fixed at 0.01 thereafter.
                epsilon = max(epsilon * args.eps_decay, args.eps_min)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                # update two neworks according to their update frequency
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                # (Exponentially Weighted Moving-Average)
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward, epsilon))
                break

    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        ## TODO ##
        with torch.no_grad():
            for t in itertools.count(start=1):

                action = agent.select_action(state, epsilon, action_space)
                # epsilon = max(epsilon * args.eps_decay, args.eps_min)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward

                if done:
                    writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                    rewards.append(total_reward)
                    print(f'Episode: {n_episode}\t\Length: {t:3d}\tTotal Reward: {total_reward:.2f}')
                    break
                
    print('Average Reward: ', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda:0')
    parser.add_argument('-m', '--model', default='dqn.pth')
    parser.add_argument('--logdir', default='log/dqn')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=2000, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=4, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--test_epsilon', default=0.05, type=float)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLander-v2')
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
