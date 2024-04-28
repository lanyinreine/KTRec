import random
from random import sample
import numpy as np
import torch
from Scripts.Agent.myModel.self_attention import encode
random.seed(2)


def create_new_feats(feature, initial_logs, initial_len):
    skill_num = feature.shape[1]
    item_num = feature.shape[0]//2
    Feature = feature.t().unsqueeze(0)
    newfeats = Feature.expand(initial_len, skill_num, item_num*2)
    for i in range(initial_len):
        line = initial_logs[i]
        newfeats[i, :, line+item_num] = newfeats[i, :, line+item_num]+feature[line]
    return newfeats


def attention_encode(x, item_num):
    batch_size, seq_len, num_nodes, num_states = x.shape
    x = x.permute(1, 0, 2, 3)
    seq_x = []
    for i in range(seq_len):
        seq_x.append(encode(x[i], item_num))
    newx = torch.stack(seq_x, dim=0)
    newx = newx.permute(1, 0, 2, 3)
    return newx


def get_initial_logs(skill_num_dict, item_types):
    initial_logs = []
    for i in range(len(item_types)):
        if i > 0:
            types = item_types[i]-item_types[i-1]
        else:
            types = item_types[i]
        initial_logs.extend(sample(skill_num_dict[types], 1))
    return initial_logs


def get_item_types(initial_len, dataset):
    item_types = []
    if dataset == "assist09":
        log_len = int(initial_len*2)
        cnt = initial_len
        s = 0
        while log_len > 0:
            types = random.randint(max(1, log_len-4*(cnt-1)), min(log_len, 4, log_len-cnt+1))
            s = s+types
            item_types.append(s)
            log_len -= types
            cnt -= 1
    elif dataset == "ednet":
        log_len = int(initial_len * 3)
        cnt = initial_len
        s = 0
        while log_len > 0:
            types = random.randint(max(1, log_len - 6 * (cnt - 1)), min(log_len, 6, log_len - cnt + 1))
            s = s + types
            item_types.append(s)
            log_len -= types
            cnt -= 1
    return item_types


def get_item_degree_list(feature, degree_list):
    item_degree_list = []
    for i in range(feature.shape[0]//2):
        item = feature[i]
        degree_item = item*degree_list
        degree_sum = torch.sum(degree_item).item()
        skill_sum = torch.sum(item).item()
        if degree_sum == 0:
            item_degree_list.append([1, 1])
        else:
            item_degree_list.append([degree_sum, skill_sum])
    return item_degree_list
