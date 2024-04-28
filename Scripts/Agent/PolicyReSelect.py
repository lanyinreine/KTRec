import torch

from KTModel.BackModels import MLP
from .ReSelect import ReSelectNetwork


class PolicyReSelect(ReSelectNetwork):
    def __init__(self, skill_num, input_size, hidden_size, pre_hidden_sizes, dropout):
        super().__init__(skill_num, input_size, hidden_size, pre_hidden_sizes, dropout)
        self.decoder2 = MLP(2 * hidden_size, pre_hidden_sizes + [skill_num], dropout=dropout)

    def policy(self, n):
        pros, paths = [], []
        targets = self.targets.squeeze(1)
        a1 = torch.arange(self.targets.size(0))
        for i in range(n):
            states = self.states[0][0]
            pro = torch.softmax(self.decoder2(torch.cat([targets.expand_as(states), states], -1)), -1)  # (B, S)
            path = torch.multinomial(pro, 1)  # (B, 1)
            # pro = torch.take_along_dim(pro, path, -1).squeeze(-1)
            pro = pro[a1, path.squeeze(-1)]
            pros.append(pro)
            paths.append(path.squeeze(-1))
            self.step(path)
        pros = torch.stack(pros, 1)
        paths = torch.stack(paths, -1)  # (B, L)
        return paths, pros

    def n_steps(self, n):
        paths1, pros1 = self.policy(n)
        paths2, pros2 = self.reselect(n, paths1)
        pros2 = torch.take_along_dim(pros2, paths2.unsqueeze(-1), -1).squeeze(-1)
        paths = torch.stack([paths1, paths2], -1)  # (B, L, 2)
        pros = torch.stack([pros1, pros2], -1)
        return paths, pros
