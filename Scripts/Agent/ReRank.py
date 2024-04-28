import torch
from torch import nn

from KTModel.BackModels import MLP, Transformer
from ..Agent.ReSelect import ReSelectNetwork
from ..Agent.utils import generate_path


class ReRankNetWork(ReSelectNetwork):
    def __init__(self, skill_num, input_size, weight_size, hidden_size, pre_hidden_sizes, dropout, allow_repeat=False,
                 withKt=False):
        super(ReRankNetWork, self).__init__(skill_num, input_size, hidden_size, pre_hidden_sizes, dropout)
        self.allow_repeat = allow_repeat
        self.withKt = withKt
        self.W1 = nn.Linear(hidden_size, weight_size, bias=False)  # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False)  # blending decoder
        self.vt = nn.Linear(weight_size, 1, bias=False)  # scaling sum of enc and dec by v.T
        # self.path_encoder = nn.TransformerEncoderLayer(hidden_size, 2, hidden_size, dropout, batch_first=True)
        self.path_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 2, hidden_size, dropout, batch_first=True), 3)
        self.decoder = nn.LSTMCell(hidden_size, hidden_size)
        self.kt_MLP = MLP(hidden_size, [hidden_size // 2, hidden_size // 4, 1], dropout=dropout)

    def rerank(self, n, origin_path=None):
        inputs = self.l2(self.embed(origin_path))
        # inputs.shape 256*20*64
        encoder_states = inputs
        if isinstance(encoder_states, tuple):
            encoder_states, _ = encoder_states
        encoder_states = self.path_encoder(encoder_states)
        # encoder_states.shape 256*20*64
        # self.targets 256*1*64
        hidden = [_.squeeze(0) for _ in self.states]
        # hidden是一个列表，两个元素，每个元素是256*64的张量
        blend1 = self.W1(encoder_states + self.targets)  # (B, L, W)
        decoder_input = torch.zeros_like(inputs[:, 0])  # (B, I)
        # decoder_input 是256*64的张量
        probs, paths = [], []
        selecting_s = []
        a1 = torch.arange(inputs.size(0))
        # a1长度256，为0到255的序列张量
        selected = torch.zeros_like(inputs[:, :, 0], dtype=torch.bool)
        minimum_fill = torch.full_like(selected, -1e9, dtype=torch.float32)
        hidden_states = []
        for i in range(n):
            hidden = self.decoder(decoder_input, hidden)
            # hidden[0]是256*64的张量
            if i > 0:
                hidden_states.append(hidden[0])
            # Compute blended representation at each decoder time step
            blend2 = self.W2(hidden[0])  # (B, W)
            blend_sum = blend1 + blend2.unsqueeze(1)  # (B, L, W)
            out = self.vt(blend_sum).squeeze(-1)  # (B, L)
            if not self.allow_repeat:
                out = torch.where(selected, minimum_fill, out)
                out = torch.softmax(out, dim=-1)
                if self.training:
                    selecting = torch.multinomial(out, 1).squeeze(-1)
                    # selecting 是一个索引,长度256
                else:
                    selecting = torch.argmax(out, 1)
                selected2 = torch.zeros_like(inputs[:, :, 0], dtype=torch.bool)
                selected2[a1, selecting] = True
                selected = selected + selected2
            else:
                out = torch.softmax(out, dim=-1)
                selecting = torch.multinomial(out, 1).squeeze(-1)
            # path = torch.take_along_dim(origin_path, selecting, -1).squeeze(-1)
            path = origin_path[a1, selecting]
            # path长度256，内容为知识编号
            decoder_input = encoder_states[a1, selecting]
            # out = torch.take_along_dim(out, selecting, -1).squeeze(-1)
            out = out[a1, selecting]
            paths.append(path)
            probs.append(out)
            selecting_s.append(selecting)
        probs = torch.stack(probs, 1)
        paths = torch.stack(paths, 1)  # (B, L)

        hidden_states.append(self.decoder(decoder_input, hidden)[0])
        hidden_states = torch.stack(hidden_states, dim=1)
        if self.withKt:
            kt_output = torch.sigmoid(self.kt_MLP(hidden_states))
            return paths, probs, kt_output
        return paths, probs

    def n_step(self, n, paths=None, path_type=0):
        if paths is None:
            paths = generate_path(self.targets.size(0), self.skill_num, path_type, n, next(self.parameters()).device)
        return self.rerank(n, paths)
