from typing import Optional, List, Callable

import math
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence


class MLP(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: List[int],
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = nn.LeakyReLU,
                 inplace: Optional[bool] = False,
                 bias: bool = True,
                 dropout: float = 0.0):
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_channels[-1], bias=bias))

        super(MLP, self).__init__(*layers)


def get_output_mask(real_len, label_len):
    batch_size = len(real_len)
    max_len = torch.max(real_len)
    seq_range_expand = torch.arange(max_len).unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = torch.unsqueeze(real_len, 1).expand_as(seq_range_expand)
    label_length_expand = seq_length_expand - label_len
    out_mask = torch.logical_and(
        torch.less(seq_range_expand, seq_length_expand),
        torch.greater_equal(seq_range_expand, label_length_expand)
    )
    return out_mask


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, hidden_sizes, dropout_rate, input_sizes=None):
        super(MultiHeadedAttention, self).__init__()
        if not isinstance(hidden_sizes, list):
            hidden_sizes = [hidden_sizes] * 4
        if input_sizes is None:
            input_sizes = hidden_sizes
        for hidden_size in hidden_sizes:
            assert hidden_size % head == 0
        self.head = head
        self.d_k = math.sqrt(hidden_sizes[0] // head)
        self.linear_s = nn.ModuleList(
            [nn.Sequential(nn.Linear(input_size, hidden_size),
                           nn.LayerNorm(hidden_size)) for (input_size, hidden_size) in zip(input_sizes, hidden_sizes)])
        self.norm = nn.LayerNorm(hidden_sizes[-1])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def attention(self, query, key, value, mask=None):
        scores = torch.div(torch.matmul(query, key.transpose(-2, -1)), self.d_k)
        if mask is not None:
            scores = scores.masked_fill(torch.logical_not(mask), -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout1(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        residual = query
        query, key, value = [l(x).view(batch_size, x.size(1), self.head, -1).transpose(1, 2)
                             for l, x in zip(self.linear_s, (query, key, value))]
        x, attn = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, x.shape[2], -1)
        x = self.dropout2(self.linear_s[-1](x))
        x += residual
        return self.norm(x)


class FeedForward(nn.Module):
    def __init__(self, head, input_size, dropout_rate):
        super(FeedForward, self).__init__()
        self.mh = MultiHeadedAttention(head, input_size, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.activate = nn.LeakyReLU()
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)

    def forward(self, s, mask):
        s = s + self.dropout1(self.mh(s, s, s, mask))
        s = self.ln1(s)
        s_ = self.activate(self.fc1(s))
        s_ = self.dropout2(self.fc2(s_))
        s = self.ln2(s + s_)
        return s


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, head=1, b=1, use_mask=True):
        super(Transformer, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.SAs = nn.ModuleList([MultiHeadedAttention(head, hidden_size, dropout_rate) for _ in range(b)])
        self.FFNs = nn.ModuleList([FeedForward(head, hidden_size, dropout_rate) for _ in range(b)])
        self.b = b
        self.use_mask = use_mask

    def forward(self, inputs):
        inputs = self.fc(inputs)
        max_len = inputs.shape[1]
        transformer_mask = None
        if self.use_mask:
            transformer_mask = torch.tril(
                torch.ones((1, max_len, max_len), dtype=torch.bool, device=inputs.device)).unsqueeze(1)
        for i in range(self.b):
            inputs = self.SAs[i](inputs, inputs, inputs, transformer_mask)
            inputs = self.FFNs[i](inputs, transformer_mask)
        return inputs


class LearnableAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.is_absolute = True
        self.embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer('position_ids', torch.arange(max_position_embeddings))

    def forward(self, x):
        """
        return (b l d) / (b h l d)
        """
        position_ids = self.position_ids[:x.size(-2)]

        return x + self.embeddings(position_ids)[None, :, :]


class DKVMN(nn.Module):
    def __init__(self, memory_size, dim_k, dim_v, hidden_size, batch_first=True):
        super(DKVMN, self).__init__()
        self.key = nn.Parameter(torch.randn((memory_size, dim_k)))
        self.E = nn.Linear(dim_v, dim_v)
        self.D = nn.Linear(dim_v, dim_v)
        self.l1 = nn.Linear(dim_k + dim_v, hidden_size)
        self.memory_size = memory_size
        self.dim_k, self.dim_v = dim_k, dim_v
        self.batch_first = batch_first

    def read(self, k_t, m_kt):
        # k_t:(B, D_K), m_kt:(B, N, D_V)
        w_t = torch.softmax(torch.matmul(k_t, self.key.transpose(1, 0)), dim=-1)  # (B, N)
        w_t = torch.unsqueeze(w_t, -1)  # (B, N, 1)
        r_t = torch.sum(w_t * m_kt, dim=1)  # (B, D_V)
        f_t = torch.tanh(self.l1(torch.cat([r_t, k_t], dim=-1)))  # (B, D_H)
        return f_t, w_t

    def write(self, v_t, w_t, m_kt):
        # v_t:(B, D_V), w_t:(B, N, 1), m_kt:(B, N, D_V)
        v_t = torch.unsqueeze(v_t, 1)  # (B, 1, D_V)
        e_t = torch.sigmoid(self.E(v_t))  # (B, 1, D_V)
        a_t = torch.tanh(self.D(v_t))  # (B, 1, D_V)
        erase = m_kt * (1 - e_t * w_t)
        adding = a_t * w_t
        return erase + adding  # (B, N, D_V)

    def get_k_v(self, q):
        if q.dim() == 3:
            return q[:, :, :self.dim_k], q
        else:
            return q[:, :self.dim_k], q

    def forward(self, q: PackedSequence, m_kt=None):
        # q: (B, L, D) or (L, B, D) after padding
        q, batch_sizes = q.data, q.batch_sizes
        batch_size, seq_len = torch.max(batch_sizes), len(batch_sizes)
        if m_kt is None:
            m_kt = torch.zeros((batch_size, self.memory_size, self.dim_v), device=q.device)  # (B, N, D_V)
        qk, qv = self.get_k_v(q)
        fs = []
        begin_loc = 0
        ts = torch.cumsum(batch_sizes, dim=-1)
        hs = []
        for t in range(seq_len):
            if m_kt.size(0) > batch_sizes[t]:
                hs.insert(0, m_kt[batch_sizes[t]:])
                m_kt = m_kt[:batch_sizes[t]]
            f_t, w_t = self.read(qk[begin_loc:ts[t]], m_kt)
            fs.append(f_t)
            m_kt = self.write(qv[begin_loc:ts[t]], w_t, m_kt)
            begin_loc = ts[t]
        hs.insert(0, m_kt)
        hs = torch.concat(hs, 0)
        fs = torch.concat(fs, dim=0)  # (L, B, D_H)
        return fs, hs


class CoKT(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, batch_first=True, head=2):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=batch_first)
        self.ma_inter = MultiHeadedAttention(head, hidden_size, dropout_rate, input_sizes=(
            hidden_size + input_size - 1, hidden_size + input_size - 1, hidden_size + input_size, hidden_size))
        self.ma_intra = MultiHeadedAttention(head, hidden_size, dropout_rate, input_sizes=(
            input_size - 1, input_size - 1, hidden_size + 1, hidden_size))
        self.wr = nn.Parameter(torch.randn(size=(1, 1, 2)))
        self.l = nn.Linear(2 * hidden_size + input_size - 1, hidden_size)

    def forward(self, intra_x, inter_his, inter_r):
        # All inputs are PackedSequence
        _, inter_his = self.rnn(inter_his)
        intra_h, _ = self.rnn(intra_x)  # (seq_sum, H)

        inter_r, batch_sizes = inter_r.data, inter_r.batch_sizes  # (seq_sum, R, I)
        intra_x, intra_h = intra_x.data, intra_h.data

        # inter attention
        inter_his = inter_his.view((-1, inter_r.size(1), inter_his.size(-1)))  # (seq_sum, R, H)
        M_rv = torch.cat([inter_his, inter_r], dim=-1)  # (seq_sum, R, H+I)
        M_pv = M_rv[:, :, :-1]  # (seq_sum, R, H+I-1)
        m_pv = torch.unsqueeze(torch.cat([intra_h, intra_x[:, :-1]], 1), dim=1)  # (seq_sum, 1, H+I-1)
        v_v = self.ma_inter(m_pv, M_pv, M_rv).squeeze(1)  # (seq_sum, H)

        # intra attention
        intra_x_p, lengths = pad_packed_sequence(PackedSequence(intra_x[:, :-1], batch_sizes),
                                                 batch_first=True)  # (bs, max_len, I-1)
        intra_h_p, _ = pad_packed_sequence(PackedSequence(torch.cat([intra_h, intra_x[:, -1:]], 1), batch_sizes),
                                           batch_first=True)  # (bs, max_len, H+1)
        # Sequence mak
        mask = torch.tril(torch.ones((1, intra_x_p.size(1), intra_x_p.size(1)), device=intra_x.device))
        v_h = self.ma_intra(intra_x_p, intra_x_p, intra_h_p, mask=mask)  # (bs, max_len, H)
        v_h = pack_padded_sequence(v_h, lengths, True).data  # (seq_sum, H)
        v = torch.sum(torch.softmax(self.wr, -1) * torch.stack((v_v, v_h), -1), -1)  # (seq_sum, H)
        return self.l(torch.cat([v, intra_h, intra_x[:, :-1]], 1))  # (seq_sum, 2*H+I-1)

    def deal_inter(self, inter_his, inter_r):
        # inter_r:(B, L, R, I)
        _, inter_his = self.rnn(inter_his)
        # inter attention
        inter_his = inter_his.view(
            (inter_r.size(0), inter_r.size(1), inter_r.size(2), inter_his.size(-1)))  # (B, L, R, H)
        M_rv = torch.cat([inter_his, inter_r], dim=-1)  # (B, L, R, H+I)
        M_pv = M_rv[:, :, :, :-1]  # (B, L, R, H+I-1)
        return M_rv, M_pv

    def step(self, M_rv, M_pv, intra_x, o, intra_h_p=None):
        # M_*: (B, R, H)
        # intra_h_p:(B, L-1, H+1), with the y
        # intra_x:(B, L, I-1), without the y
        # o: y from last step
        intra_h_next, _ = self.rnn(torch.cat([intra_x[:, -1:], o], dim=-1),
                                   None if intra_h_p is None else intra_h_p[:, -1, :-1].unsqueeze(
                                       0).contiguous())  # (B, 1, H)
        m_pv = torch.cat([intra_h_next, intra_x[:, -1:]], -1)  # (B, 1, H+I-1)
        v_v = self.ma_inter(m_pv, M_pv, M_rv)  # (B, 1, H)

        intra_x_p = intra_x
        intra_h_next = torch.cat([intra_h_next, o], dim=-1)
        intra_h_p = intra_h_next if intra_h_p is None else torch.cat([intra_h_p, intra_h_next], dim=1)  # (B, L, H+1)
        # Sequence mask
        v_h = self.ma_intra(intra_x_p[:, -1:], intra_x_p, intra_h_p)  # (B, 1, H), only query last target item
        v = torch.sum(torch.softmax(self.wr, -1) * torch.stack((v_v, v_h), -1), -1)  # (B, 1, H)
        return self.l(torch.cat((v, intra_h_p[:, -1:, :-1], intra_x[:, -1:]), -1)), intra_h_p  # (B, 1, 2*H+I-1)
