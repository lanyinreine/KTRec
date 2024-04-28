import torch
from torch import nn
from KTModel.BackModels import MLP, Transformer, LearnableAbsolutePositionEmbedding


class ReSelectNetwork(nn.Module):
    def __init__(self, skill_num, input_size, hidden_size, pre_hidden_sizes, dropout):
        super().__init__()
        self.skill_num = skill_num
        self.l1 = nn.Linear(input_size + 1, input_size)
        self.l2 = nn.Linear(input_size, hidden_size)
        self.embed = nn.Embedding(skill_num, input_size)
        self.state_encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.path_encoder = Transformer(hidden_size, hidden_size, dropout, head=1, b=1, use_mask=False)
        self.pos = LearnableAbsolutePositionEmbedding(200, input_size)
        self.decoder = MLP(hidden_size, pre_hidden_sizes + [skill_num], dropout=dropout)
        self.targets = None
        self.states = None
        self.hidden_size = hidden_size

    def begin_episode(self, targets, initial_logs=None):
        # targets: (B, K), where K is the num of targets in this batch
        # initial_logs: (B, IL, 2)
        self.states = (torch.zeros(1, targets.size(0), self.hidden_size).to(targets.device),
                       torch.zeros(1, targets.size(0), self.hidden_size).to(targets.device))
        self.targets = torch.mean(self.embed(targets), dim=1, keepdim=True)  # (B, 1, I)
        self.targets = self.l2(self.targets)
        if initial_logs is not None:
            if len(initial_logs.shape) == 3:
                self.step(initial_logs[:, :, 0], initial_logs[:, :, 1])
            else:
                self.step(initial_logs)

    def step(self, x, score=None):
        x = self.embed(x.long())

        if score is not None:
            x = self.l1(torch.cat([x, score.unsqueeze(-1)], -1))
        # states为元组，包含两个1*256*64的张量
        # x为256*10*48的张量
        _, self.states = self.state_encoder(x, self.states)
        # self.states = torch.mean(self.encoder(x), dim=1, keepdim=True)

    def reselect(self, n, path=None):
        if path is None:
            path = torch.randint(self.skill_num, (self.targets.size(0), n)).to(self.l1.device)
        # inputs = self.l2(self.pos(self.embed(path)))
        # states, _ = self.encoder(inputs, self.states)  # (B, L, H)
        # states = self.encoder2(inputs)
        inputs = self.l2(self.embed(path))
        inputs = inputs + torch.mean(inputs, dim=1, keepdim=True)
        states = inputs + self.states[0].squeeze().unsqueeze(1)
        states = states + self.targets
        pro = torch.softmax(self.decoder(states), -1)  # (B, L, S)
        path = torch.multinomial(pro.view(-1, self.skill_num), 1).view(path.shape)
        return path, pro

    def n_steps(self, n):
        path, pro = self.reselect(n)
        a1 = torch.arange(pro.size(0)).unsqueeze(1).repeat_interleave(dim=1, repeats=pro.size(1))
        a2 = torch.arange(pro.size(1)).unsqueeze(0).repeat_interleave(dim=0, repeats=pro.size(0))
        pro = pro[a1, a2, path]
        # pro = torch.take_along_dim(pro, path.unsqueeze(-1), -1).squeeze(-1)
        return path, pro


if __name__ == '__main__':
    device = 'cpu'
    agent = ReSelectNetwork(10, 24, 32, [64, 16, 4], 0.5)
    agent = agent.to(device)
    targets_ = torch.randint(10, size=(2, 3)).to(device)
    initial_logs_ = torch.randint(10, size=(2, 6, 2)).float()
    initial_logs_[:, :, -1] = torch.rand(size=(2, 6))
    initial_logs_ = initial_logs_.to(device)
    agent.begin_episode(targets_, initial_logs_)
    print(agent.n_steps(10))
