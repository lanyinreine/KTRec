import torch
from torch import nn

from KTModel.BackModels import MLP, Transformer
from Scripts.Agent.myModel.TGCN.TGCNReselect import TGCNReSelectNetwork
from Scripts.Agent.utils import generate_path
from Scripts.Agent.myModel.TGCN.tgcn import TGCNCell


class TGCNReRankNetWork(TGCNReSelectNetwork):
    def __init__(self, skill_num, input_size, weight_size, hidden_size, pre_hidden_sizes, dropout, adj, allow_repeat=False,
                 withKt=False):
        super(TGCNReRankNetWork, self).__init__(skill_num, input_size, hidden_size, pre_hidden_sizes, dropout, adj)
        self.allow_repeat = allow_repeat
        self.withKt = withKt
        self.W1 = nn.Linear(hidden_size, weight_size, bias=False)  # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False)  # blending decoder
        self.W3 = nn.Linear(hidden_size, 1, bias=False)  # blending decoder
        self.vt = nn.Linear(weight_size, 1, bias=False)  # scaling sum of enc and dec by v.T
        # self.path_encoder = nn.TransformerEncoderLayer(hidden_size, 2, hidden_size, dropout, batch_first=True)
        self.path_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 2, hidden_size, dropout, batch_first=True), 3)
        # self.decoder = nn.LSTMCell(hidden_size, hidden_size)
        self.decoder = TGCNCell(adj, adj.shape[0], hidden_size, hidden_size)
        self.kt_MLP = MLP(hidden_size, [hidden_size // 2, hidden_size // 4, 1], dropout=dropout)
    def rerank(self, n, origin_path=None):
        inputs = self.l2(self.embed(origin_path))
        encoder_states = inputs
        if isinstance(encoder_states, tuple):
            encoder_states, _ = encoder_states
        encoder_states = self.path_encoder(encoder_states)
        hidden = self.states
        blend1 = self.W1(encoder_states + self.targets)  # (B, L, W)
        decoder_input = torch.zeros_like(inputs[:, 0])  # (B, H)
        probs, paths = [], []
        selecting_s = []
        a1 = torch.arange(inputs.size(0))
        selected = torch.zeros_like(inputs[:, :, 0], dtype=torch.bool)
        minimum_fill = torch.full_like(selected, -1e9, dtype=torch.float32)
        # KT用
        hidden_states = []
        skill_probs = []
        batch_size, hidden_dim = decoder_input.shape
        num_nodes = len(self.adj[0])
        tgcncell_hidden_state = None

        for i in range(n):
            hidden = hidden+decoder_input.unsqueeze(1)


            tgcncell_hidden_state = torch.zeros(batch_size, num_nodes * hidden_dim).type_as(hidden)
            hidden, _ = self.decoder(hidden, tgcncell_hidden_state)
            hidden = hidden.reshape(batch_size, num_nodes, hidden_dim)
            # hidden:256*123*64

            if i > 0:
                hidden_states.append(hidden)

            # Compute blended representation at each decoder time step
            blend2 = self.W2(hidden)  # 256*123*64
            # blend1:256*20*64
            blend_sum = blend1 + blend2  # (B, L, W)
            out = self.vt(blend_sum).squeeze(-1)  # (B, L)
            # out:256*20
            if not self.allow_repeat:
                out = torch.where(selected, minimum_fill, out)
                out = torch.softmax(out, dim=-1)
                if self.training:
                    selecting = torch.multinomial(out, 1).squeeze(-1)
                    # selecting: 256,0到19的索引
                else:
                    selecting = torch.argmax(out, 1)
                selected2 = torch.zeros_like(inputs[:, :, 0], dtype=torch.bool)
                # selected2:256*20
                selected2[a1, selecting] = True
                selected = selected + selected2
            else:
                out = torch.softmax(out, dim=-1)
                selecting = torch.multinomial(out, 1).squeeze(-1)
            # path = torch.take_along_dim(origin_path, selecting, -1).squeeze(-1)
            path = origin_path[a1, selecting]
            decoder_input = encoder_states[a1, selecting]
            # out = torch.take_along_dim(out, selecting, -1).squeeze(-1)
            skill_probs.append(out)
            out = out[a1, selecting]
            # print("shapes", path.shape, decoder_input.shape, out.shape)
            paths.append(path)
            probs.append(out)
            selecting_s.append(selecting)
        probs = torch.stack(probs, 1)
        paths = torch.stack(paths, 1)  # (B, L)
        skill_probs = torch.stack(skill_probs, dim=1)
        # 加入KT
        hidden_states_1, tmp = self.decoder(hidden, tgcncell_hidden_state)
        hidden_states_1 = hidden_states_1.reshape(batch_size, num_nodes, hidden_dim)
        hidden_states.append(hidden_states_1)
        hidden_states = torch.stack(hidden_states, dim=1)
        if self.withKt:
            tmp = self.kt_MLP(hidden_states)
            kt_output = torch.sigmoid(self.kt_MLP(hidden_states))
            # 把kt_output从256.20.123.1变到256.20.1，要么改MLP
            kt_output = torch.squeeze(kt_output, dim=-1)
            kt_output = torch.mean(kt_output,dim=-1,keepdim=True)
            return paths, probs, kt_output, skill_probs
            # return paths, probs, kt_output

        return paths, probs

    def n_step(self, n, paths=None, path_type=0):
        if paths is None:
            paths = generate_path(self.targets.size(0), self.skill_num, path_type, n, next(self.parameters()).device)
        return self.rerank(n, paths)
