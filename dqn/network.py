from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from bit_switch_env import BitSwitchEnv


class DQN(nn.Module):
    def __init__(self, n_states: int, n_actions: int, n_hidden: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.qnet = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
        )

    def forward(self, x):
        return self.qnet(x)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ExperienceBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque([], maxlen=capacity)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def add(self, *args):
        self.buffer.append(Transition(*args))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, sample_size: int, epsilon = 0.01, lr = 0.01, gamma = 0.99) -> None:

        self.epsilon = epsilon
        self.sample_size = sample_size
        self.gamma = gamma

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim, 128)
        self.target_net = DQN(state_dim, action_dim, 128)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.exp_buffer = ExperienceBuffer(int(1e4))

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def pick_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def compute_loss(self):
        self.optimizer.zero_grad()

        batch = self.exp_buffer.sample(self.sample_size)

        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        pre_q_values = self.policy_net(states)
        
        q_values = pre_q_values.gather(1, actions)

        next_q_values = self.target_net(next_states).max(1)[0].detach()

        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values.squeeze(), targets)

        return loss


if __name__ == '__main__':
    model = DQNAgent(4, 4, 100)
    env = BitSwitchEnv(4)
    goal = env.reset()
    state = env.state

    for epoch in range(1000):
        while not model.exp_buffer.is_full():
            action = model.pick_action(torch.from_numpy(state))

            prev_state = np.array(state, copy=True)
            state, reward, done = env.step(action)

            model.exp_buffer.add(prev_state, [action], reward, state, done)

            if env.reached:
                goal = env.reset()
                state = env.state

        # done with the epoch
        loss = model.compute_loss()
        loss.backward()

        model.optimizer.step()

        goal = env.reset()
        state = env.state
        model.exp_buffer.reset()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
        if (epoch + 1) == 800:
            print("hi")
