import torch
from torch import nn
from KTModel.BackModels import MLP, Transformer, LearnableAbsolutePositionEmbedding
from Scripts.Agent.myModel.TGCN.tgcn import TGCN


class TGCNReSelectNetwork(nn.Module):
    def __init__(self, skill_num, input_size, hidden_size, pre_hidden_sizes, dropout, adj):
        super().__init__()
        self.skill_num = skill_num
        self.l1 = nn.Linear(input_size + 1, input_size)
        self.l2 = nn.Linear(input_size, hidden_size)
        self.embed = nn.Embedding(skill_num, input_size)
        # input_size=123,hidden_size=64
        # self.state_encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.state_encoder = TGCN(adj, hidden_size, 64)
        # self.state_decoder = TGCNCell(adj, adj.shape[0], hidden_size)
        self.path_encoder = Transformer(hidden_size, hidden_size, dropout, head=1, b=1, use_mask=False)
        self.pos = LearnableAbsolutePositionEmbedding(200, input_size)
        self.decoder = MLP(hidden_size, pre_hidden_sizes + [skill_num], dropout=dropout)
        self.targets = None
        self.states = None
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.adj = adj

    def begin_episode(self, targets, x):
        self.targets = torch.mean(self.embed(targets), dim=1, keepdim=True)  # (B, 1, I)
        self.targets = self.l2(self.targets)
        self.states = self.state_encoder(x)
        # self.states是一个256*123*64的张量
        # hidden_state = torch.zeros(batch_size, num_nodes * self.hidden_size).type_as(self.states)
        # self.states = self.state_decoder(self.states, hidden_state)

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

