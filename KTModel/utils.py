import random
from argparse import Namespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pack_sequence, PackedSequence

from KTModel.PredictModel import PredictRetrieval, PredictModel


def collate_fn(data_):
    data_ = [_ for _ in data_ if len(_[1]) > 1]
    data_.sort(key=lambda _: len(_[1]), reverse=True)
    users = torch.as_tensor([_[0] for _ in data_])  # (bs,)
    logs_packed = pack_sequence([torch.as_tensor(_[1], dtype=torch.int) for _ in data_])  # (bs,max_len,2) after pad
    y = logs_packed.data[:, -1].float()
    return users, logs_packed, y


def collate_rec(data_):
    users, logs_packed, y = collate_fn(data_)
    logs_packed = PackedSequence(logs_packed.data[:, :-1], logs_packed.batch_sizes)
    return users, logs_packed, y.long()


def collate_co(data_):
    users, logs_packed, y = collate_fn(data_)
    r_his = [_ for l in data_ for _ in l[-2]]  # bs*(seq_len*R)个相似学生序列,seq_len可变
    r_his = pack_sequence([torch.tensor(_, dtype=torch.int32) for _ in r_his], enforce_sorted=False)
    r_skill_y = pack_sequence(([torch.tensor(_[-1], dtype=torch.int32) for _ in data_]))  # (bs,max_len,R,2) after pad
    return users, logs_packed, r_his, r_skill_y, y


def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"random seed set to be {seed}")


def load_model(args):
    if isinstance(args, dict):
        args = Namespace(**args)
    with_label = not args.without_label
    if args.model in ['DKT', 'DKVMN', 'Transformer', 'GRU4Rec']:
        return PredictModel(
            feat_nums=args.feat_nums,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.pre_hidden_sizes,
            dropout=args.dropout,
            output_size=args.output_size,
            with_label=with_label,
            model_name=args.model)
    elif args.model in ['CoKT']:
        return PredictRetrieval(
            feat_nums=args.feat_nums,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.pre_hidden_sizes,
            dropout=args.dropout,
            with_label=with_label,
            model_name=args.model)
    else:
        raise NotImplementedError


def evaluate_utils(y_, y, criterion=None):
    if criterion is not None:
        loss = criterion(y_, y)
    else:
        loss = None
    y_ = y_.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y_shape = y_.shape
    acc = np.mean((np.argmax(y_, -1) if len(y_shape) > 1 else y_ > 0.5) == y)
    auc = roc_auc_score(y_true=y, y_score=y_) if len(y_shape) == 1 else acc
    return loss, acc, auc
