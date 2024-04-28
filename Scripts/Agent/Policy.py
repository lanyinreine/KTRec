import torch

from .ReSelect import ReSelectNetwork
from KTModel.BackModels import MLP


class PolicyNetwork(ReSelectNetwork):
    def __init__(self, skill_num, input_size, hidden_size, pre_hidden_sizes, dropout):
        super().__init__(skill_num, input_size, hidden_size, pre_hidden_sizes, dropout)
        self.decoder = MLP(2 * hidden_size, pre_hidden_sizes + [skill_num], dropout=dropout)

    def begin_episode(self, targets, initial_logs=None):
        super(PolicyNetwork, self).begin_episode(targets, initial_logs)
        self.targets = self.targets.squeeze(1)

    def n_steps(self, n):
        pros, paths = [], []
        a1 = torch.arange(self.targets.size(0))
        for i in range(n):
            states = self.states[0][0]
            pro = torch.softmax(self.decoder(torch.cat([self.targets.expand_as(states), states], -1)), -1)  # (B, S)
            path = torch.multinomial(pro, 1)  # (B, 1)
            # pro = torch.take_along_dim(pro, path, -1).squeeze(-1)
            pro = pro[a1, path.squeeze(-1)]
            pros.append(pro)
            paths.append(path.squeeze(-1))
            self.step(path)
        pros = torch.stack(pros, 1)
        paths = torch.stack(paths, -1)  # (B, L)
        return paths, pros
