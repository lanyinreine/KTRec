import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

from KTModel.BackModels import MLP, DKVMN, Transformer, CoKT


class PredictModel(nn.Module):
    def __init__(self, feat_nums, input_size, hidden_size, pre_hidden_sizes, dropout, output_size=1, with_label=True,
                 model_name='DKT'):
        super(PredictModel, self).__init__()
        self.item_embedding = nn.Embedding(feat_nums, input_size)
        self.mlp = MLP(hidden_size, pre_hidden_sizes + [output_size], dropout=dropout,norm_layer=nn.BatchNorm1d)
        self.with_label = with_label
        self.move_label = True
        input_size_label = input_size + 1 if with_label else input_size
        if model_name == 'DKT':
            self.rnn = nn.LSTM(input_size_label, hidden_size, batch_first=True)
        elif model_name == 'DKVMN':
            self.rnn = DKVMN(feat_nums, input_size, input_size_label, hidden_size, True)
        elif model_name == 'Transformer':
            self.rnn = Transformer(input_size_label, hidden_size, dropout, head=8, b=1)
        elif model_name == 'GRU4Rec':
            self.rnn = nn.GRU(input_size_label, hidden_size, batch_first=True)
            self.move_label = False

    def forward(self, x: PackedSequence):
        # x:(len_sum,2)
        x, batch_sizes = x.data, x.batch_sizes
        x, y = x[:, 0], x[:, 1:]
        x = self.item_embedding(x.long())
        if self.with_label:
            if self.move_label:
                y, lengths = pad_packed_sequence(PackedSequence(y, batch_sizes), batch_first=True)
                y = torch.cat([torch.zeros_like(y[:, 0:1]), y[:, :-1]], dim=1).float()
                y = pack_padded_sequence(y, lengths, True).data
            x = torch.cat([x, y], dim=-1)
        x = PackedSequence(x, batch_sizes)
        x, lengths = pad_packed_sequence(x, True)
        o = self.rnn(x)
        if isinstance(o, tuple) and not isinstance(o, PackedSequence):
            o = o[0]
        o = pack_padded_sequence(o, lengths, True)
        if isinstance(o, PackedSequence):
            o = o.data
        o = self.mlp(o)
        if o.size(-1) == 1:
            o = torch.sigmoid(o)
        else:
            o = torch.softmax(o, -1)
        return o

    @torch.no_grad()
    def learn(self, x, states=None, get_score=True):
        # x:256*10
        seq_len = x.size(1)
        x = self.item_embedding(x.long())
        # x:256*10*128
        o = torch.zeros_like(x[:, 0:1, 0:1])
        # o:256*1*1
        os = []
        for i in range(seq_len):
            x_i = x[:, i:i + 1]
            # x_i=256*1*128
            if self.with_label and get_score:
                x_i = torch.cat([x_i, o], dim=-1)
            # x_i=256*1*129
            o, states = self.rnn(x_i, states)
            # o=256*1*128,states 长度2元组，每个元素1*256*128
            if get_score:
                o = torch.sigmoid(self.mlp(o.squeeze(1))).unsqueeze(1)
            # o=256*1*1
            os.append(o)
        o = torch.cat(os, dim=1)  # (B, L, 1) or (B, L, H)
        # o=256*seq_len*1
        return o, states

    def take_embed(self, targets):
        targets = torch.mean(self.item_embedding(targets), dim=1, keepdim=True)
        return targets


class PredictRetrieval(PredictModel):
    def __init__(self, feat_nums, input_size, hidden_size, pre_hidden_sizes, dropout, with_label=True,
                 model_name='CoKT'):
        super(PredictRetrieval, self).__init__(feat_nums, input_size, hidden_size, pre_hidden_sizes, dropout, 1,
                                               with_label, model_name)
        if model_name == 'CoKT':
            self.rnn = CoKT(input_size + 1, hidden_size, dropout, batch_first=True, head=2)

    def forward(self, x):
        intra_x, inter_his, inter_r = x

        intra_x, batch_sizes = intra_x.data, intra_x.batch_sizes
        intra_x, y = intra_x[:, 0], intra_x[:, 1:]
        intra_x = self.item_embedding(intra_x.long())
        y, lengths = pad_packed_sequence(PackedSequence(y, batch_sizes), batch_first=True)
        y = torch.cat([torch.zeros_like(y[:, 0:1]), y[:, :-1]], dim=1).float()
        y = pack_padded_sequence(y, lengths, True).data
        intra_x = torch.cat([intra_x, y], dim=-1)
        intra_x = PackedSequence(intra_x, batch_sizes)

        inter_his = PackedSequence(
            torch.cat([self.item_embedding(inter_his.data[:, 0].long()), inter_his.data[:, 1:].float()], 1),
            inter_his.batch_sizes)
        inter_r = PackedSequence(
            torch.cat([self.item_embedding(inter_r.data[:, :, 0].long()), inter_r.data[:, :, 1:].float()], -1),
            inter_r.batch_sizes)
        o = self.rnn(intra_x, inter_his, inter_r)
        o = torch.sigmoid(self.mlp(o))
        return o

    @torch.no_grad()
    def learn(self, x, states=None):
        # states:All history of intra_x and intra_h:(B, L_H, I), (B, L_H, H)
        intra_x, inter_his, inter_r = x  # intra_x:(B, L), inter_r:(B, L, R, 2)
        his_len, seq_len = 0, intra_x.size(1)
        intra_x = self.item_embedding(intra_x.long())  # (B, L, I)
        intra_h = None
        if states is not None:
            his_len = states[0].size(1)
            intra_x = torch.cat([intra_x, states[0]], 1)  # (B, L_H+L, I)
            intra_h = states[1]
        o = torch.zeros_like(intra_x[:, 0:1, 0:1])

        inter_his = PackedSequence(
            torch.cat([self.item_embedding(inter_his.data[:, 0].long()), inter_his.data[:, 1:].float()], 1),
            inter_his.batch_sizes)
        inter_r = torch.cat([self.item_embedding(inter_r[:, :, :, 0].long()), inter_r[:, :, :, 1:].float()], -1)

        M_rv, M_pv = self.rnn.deal_inter(inter_his, inter_r)  # (B, L, R, H)
        os = []
        for i in range(seq_len):
            o, intra_h = self.rnn.step(M_rv[:, i], M_pv[:, i], intra_x[:, :i + his_len + 1], o, intra_h)
            o = torch.sigmoid(self.mlp(o))
            os.append(o)
        o = torch.cat(os, dim=1)  # (B, L, 1)
        return o, (intra_x, intra_h)
