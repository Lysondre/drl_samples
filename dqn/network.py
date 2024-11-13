from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


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

class ExperienceBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

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
        batch = self.exp_buffer.sample(self.sample_size)

        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, actions)

        next_q_values = self.target_net(next_states).max(1)[0].detach()

        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values.squeeze(), targets)

        return loss


if __name__ == '__main__':
    model = DQN(3, 2, 128)
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    x_train = torch.tensor([[i] for i in range(-10, 11)], dtype=torch.float32)
    y_train = x_train ** 2

    for epoch in range(10000):
        model.train()

        y_pred = model(x_train)

        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

    x = torch.tensor([3], dtype=torch.float32)

    y = model(x)

    print(y)
