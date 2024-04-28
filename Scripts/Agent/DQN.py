import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from Scripts.Agent.utils import generate_path


class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.action_dim = action_dim
        self.embed = nn.Embedding(action_dim, state_dim)
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, actions=None):
        if actions is None:
            actions = self.embed.weight.unsqueeze(0)
        else:
            actions = self.embed(actions)
        x = x.unsqueeze(1) + actions  # (B, A, S)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)  # (B, A)


class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon=0.99, target_update=10,
                 device='cpu'):
        super().__init__()
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.targets = None
        self.states = None
        self.d_model = None
        self.device = device

    def n_steps(self, h, n, path_type=0, step_by_step=False):
        if self.epsilon > 0.1:
            self.epsilon *= 0.99
        a1 = torch.arange(h.size(0))
        actions = generate_path(h.size(0), self.action_dim, path_type, n, next(self.parameters()).device)
        if self.training and np.random.random() < self.epsilon:
            pass
        else:
            if not step_by_step:
                actions_arg = torch.argsort(self.q_net(h, actions)[:, :n], -1, descending=True)
                actions = actions[a1, actions_arg]
            else:
                selected = torch.zeros_like(actions, dtype=torch.bool)
                hs = []
                actions_sorted = []
                for i in range(n):
                    v = self.q_net(h, actions)
                    v[selected] = -500
                    action = torch.argmax(v, -1)  # (B,)
                    selected[a1, action] = True
                    action = actions[a1, action]
                    actions_sorted.append(action)
                    h, self.states = self.d_model.learn(action.unsqueeze(-1), self.states, get_score=False)
                    h = h[:, -1] + self.targets
                    hs.append(h)
                actions = torch.stack(actions_sorted, 1)
                hs = torch.stack(hs, 1)
                return actions, hs
        actions = actions[:, :n]
        hs, self.states = self.d_model.learn(actions, self.states, get_score=False)
        return actions, hs

    def get_d_model(self, d_model):
        self.d_model = d_model

    def begin_episode(self, targets, initial_logs):
        self.targets = torch.mean(self.d_model.take_embed(targets), dim=1)
        h, self.states = self.d_model.learn(initial_logs, get_score=False)
        h = h[:, -1] + self.targets
        return h, self.states

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float, device=self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64, device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float, device=self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float, device=self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float, device=self.device).view(-1, 1)

        q_values = self.q_net(states, actions).view(-1, 1)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        return dqn_loss
