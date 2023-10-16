'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time
import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from atari_wrappers import wrap_deepmind, make_atari
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if gpu is to be used

class ReplayMemory(object):
    ## TODO ##
    def __init__(self, capacity, state_shape, n_actions, device):
        c,h,w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, done):
        """Saves a transition"""
        self.m_states[self.position] = state # 5,84,84
        self.m_actions[self.position,0] = action
        self.m_rewards[self.position,0] = reward
        self.m_dones[self.position,0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)
        
    def sample(self, bs):
        """Sample a batch of transitions"""
        i = torch.randint(0, high=self.size, size=(bs,))
        bs = self.m_states[i, :4] 
        bns = self.m_states[i, 1:] 
        ba = self.m_actions[i].to(self.device)
        br = self.m_rewards[i].to(self.device).float()
        bd = self.m_dones[i].to(self.device).float()
        return bs, ba, br, bns, bd
        
    def __len__(self):
        return self.size


class Net(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.to(device)
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)


def fp(n_frame):
    n_frame = torch.from_numpy(n_frame)
    h = n_frame.shape[-2]
    return n_frame.view(1,h,h)

class FrameProcessor():
    def __init__(self, im_size=84):
        self.im_size = im_size

    def process(self, frame):
        im_size = self.im_size
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[46:160+46, :]

        frame = cv2.resize(frame, (im_size, im_size), interpolation=cv2.INTER_LINEAR)
        frame = frame.reshape((1, im_size, im_size))

        x = torch.from_numpy(frame)
        return x

h,w = 84,84
# Action [4]: 0 (No-op), 1 (Fire), 2 (Right), 3 (Left)
n_actions = 4

# c,h,w = m.fp(env.reset()).shape
# n_actions = env.action_space.n
M_SIZE = 100000

class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._target_net.eval()
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(), lr=args.lr, eps=1.5e-4)

        ## TODO ##
        """Initialize replay buffer"""
        #self._memory = ReplayMemory(...)
        self._memory = ReplayMemory(M_SIZE, [5,h,w], n_actions, args.device)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        # sa = m.ActionSelector(eps, eps, policy_net, EPS_DECAY, n_actions, device)
        ## not sure
        ## TODO ##
        if random.random() < epsilon:
            a = torch.tensor([[random.randrange(n_actions)]], device='cpu', dtype=torch.long)
        else:
            with torch.no_grad():
                a = self._behavior_net(state).max(1)[1].cpu().view(1,1)

        return a.numpy()[0,0].item()

    def append(self, state, action, reward, done):
        ## TODO ##
        """Push a transition into replay buffer"""
        self._memory.push(state, action, reward, done) 

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size)
        ## TODO ##
        q = self._behavior_net(state).gather(1, action)
        nq = self._target_net(next_state).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (nq * gamma)*(1.-done[:,0]) + reward[:,0]

        # Compute Huber loss
        loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

        # criterion = nn.MSELoss()
        # loss = criterion(q, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._behavior_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()
        

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        

    def save(self, model_path, checkpoint=True):
        if checkpoint:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
                'target_net': self._target_net.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=True):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])

def train(args, agent, writer):
    print('Start Training')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)
    # env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)

    c,h,w = fp(env.reset()).shape
    n_actions = env.action_space.n


    action_space = env.action_space
    total_steps, epsilon = 1, 1.
    ewma_reward = 0
    q = deque(maxlen=5)
    done = True

    for episode in range(args.episode):

        total_reward = 0
        state = env.reset()
        state, reward, done, _ = env.step(1) # fire first !!!
        for i in range(10): # no-op
            n_frame, _, _, _ = env.step(0)
            n_frame = fp(n_frame)
            q.append(n_frame)

        for t in itertools.count(start=1):

            # Select and perform an action
            state = torch.cat(list(q))[1:].unsqueeze(0)

            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                # decay epsilon
                epsilon -= (1 - args.eps_min) / args.eps_decay
                epsilon = max(epsilon, args.eps_min)

            
            n_frame, reward, done, info = env.step(action)
            n_frame = fp(n_frame)

            # 5 frame as memory
            q.append(n_frame)
            # memory.push(torch.cat(list(q)).unsqueeze(0), action, reward, done) # here the n_frame means next frame from the previous time step
            agent.append(torch.cat(list(q)).unsqueeze(0), action, reward, done)

            if total_steps >= args.warmup:
                agent.update(total_steps)

            total_reward += reward

            if total_steps % args.eval_freq == 0:
                print("total_steps = ",total_steps)
                print("args.eval_freq = ",args.eval_freq)

                """You can write another evaluate function, or just call the test function."""
                test(args, agent, writer)
                # evaluate(step, policy_net, device, env_raw, n_actions, eps=0.05, num_episode=15)
                agent.save(args.model + "dqn_b_" + str(total_steps) + ".pt")

            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward, episode)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, episode)
                print('Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                        .format(total_steps, episode, t, total_reward, ewma_reward, epsilon))
                break
    env.close()


def test(args, agent, writer):
    print('Start Testing')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    # Remember to set episode_life=False, clip_rewards=False while testing.
    env = wrap_deepmind(env_raw,episode_life=False, clip_rewards=False)
    action_space = env.action_space
    e_rewards = []
    q = deque(maxlen=5)
    
    for i in range(args.test_episode):
        state = env.reset()
        e_reward = 0
        for _ in range(10): # no-op
            n_frame, _, done, _ = env.step(0)
            n_frame = fp(n_frame)
            q.append(n_frame)

        done = False

        while not done:
            # time.sleep(0.01)
            # env.render()
            state = torch.cat(list(q))[1:].unsqueeze(0)
            action = agent.select_action(state, args.test_epsilon, action_space)
            n_frame, reward, done, info = env.step(action)
            n_frame = fp(n_frame)
            q.append(n_frame)
            e_reward += reward

        writer.add_scalar('Test/Episode Reward', e_reward, i+1)
        e_rewards.append(e_reward)
        print('episode {}: {:.2f}'.format(i+1, e_reward))
        
    env.close()
    print('Average Reward: {:.2f}'.format(float(sum(e_rewards)) / float(args.test_episode)))


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='ckpt/')
    parser.add_argument('--logdir', default='log/dqn_b')
    # train
    parser.add_argument('--warmup', default=20000, type=int)
    parser.add_argument('--episode', default=200000, type=int)
    parser.add_argument('--capacity', default=50000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0000625, type=float)
    parser.add_argument('--eps_decay', default=500000, type=float)
    parser.add_argument('--eps_min', default=0.1, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=20000, type=int)
    parser.add_argument('--eval_freq', default=200000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('-tmp', '--test_model_path', default='ckpt/dqn_b_43000000.pt')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--test_episode', default=10, type=int)
    parser.add_argument('--seed', default=20230422, type=int)
    parser.add_argument('--test_epsilon', default=0.05, type=float)
    args = parser.parse_args()

    ## main ##
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if args.test_only:
        agent.load(args.test_model_path)
        test(args, agent, writer)
    else:
        train(args, agent, writer)
        agent.save(args.model+ "dqn_b_final" + ".pt")
        


if __name__ == '__main__':
    main()
