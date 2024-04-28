import random
from random import sample
import numpy as np
import torch
from Scripts.Agent.myModel.self_attention import encode
random.seed(2)


def create_new_feats(feature, initial_logs, initial_len):
    Feature = feature.t().unsqueeze(0)
    newfeats = Feature.expand(initial_len, 140, 909*2)
    for i in range(initial_len):
        line = initial_logs[i]
        newfeats[i,:,line+909] = newfeats[i,:,line+909]+feature[line]
    return newfeats

# 改之前的！！！！！！！！！！！
# def attention_encode(x):
#     batch_size, seq_len, num_nodes, num_states = x.shape
#
#     batch_x = []
#     for i in range(batch_size):
#         seq_x = []
#         for j in range(seq_len):
#             seq_x.append(encode(x[i][j]).unsqueeze(0))
#         seq_x = torch.cat(seq_x, dim=0)
#         batch_x.append(seq_x.unsqueeze(0))
#     return torch.cat(batch_x, dim=0)
def attention_encode(x):

    batch_size, seq_len, num_nodes, num_states = x.shape
    x = x.permute(1, 0, 2, 3)
    seq_x = []
    for i in range(seq_len):
        seq_x.append(encode(x[i]))
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


def get_item_types(initial_len):
    item_types = []
    log_len = int(initial_len*3)
    cnt = initial_len
    s = 0
    while log_len > 0:
        types = random.randint(max(1,log_len-6*(cnt-1)), min(log_len, 6, log_len-cnt+1))
        s = s+types
        item_types.append(s)
        log_len -= types
        cnt -= 1
    return item_types


"""
def get_initial_logs(skill_num_dict):
    lengths = [0, 4, 3, 2, 1]
    initial_logs = []
    for types in range(1, 5):
        initial_logs.extend(sample(skill_num_dict[types], lengths[types]))
    return initial_logs
"""

"""
def get_model(embed_size):
    input_size = 1
    output_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelfAttentionEncoder(input_size, output_size).to(device)
    model.load_state_dict(torch.load('D:\\data\\model\\trained_attention.pth'))
    return model
"""


def get_item_degree_list(feature, degree_list):
    item_degree_list = []
    for i in range(948):
        item = feature[i]
        degree_item = item*degree_list
        degree_sum = torch.sum(degree_item).item()
        skill_sum = torch.sum(item).item()
        if degree_sum == 0:
            item_degree_list.append([1, 1])
        else:
            item_degree_list.append([degree_sum, skill_sum])
    return item_degree_list

def get_adj_dict(adj):
    adj_list = adj.tolist()
    adj_dict = {}
    for i in range(len(adj_list[0])):
        adj_dict[i] = [i]
        for j in range(len(adj_list[i])):
            if adj_list[i][j]==1:
                adj_dict[i].append(j)
    return adj_dict

def get_item_path(paths, skill_probs, adj_dict, degree_list):
    # paths 256*20
    # skill_probs 256*20*123
    item_paths = []
    all_skill = [i for i in range(140)]
    batch_size, seq_len = paths.shape
    for batch in range(batch_size):
        item_path = []
        for time in range(seq_len):
            skill = paths[batch][time].item()
            skills_selecting = adj_dict[skill]
            probs = skill_probs[batch][time]
            # item_selected = select_item(skills_selecting, probs, degree_list)
            item_selected = select_item(all_skill, probs, degree_list)
            # print(item_selected)

def select_item(skills, probs, degree_list):
    length = len(skills)
    if length == 1:
        return skills
    item_probs = torch.zeros(length**2)
    for i in range(length**2):
        a, b = i//length, i%length
        if a==b:
            continue
        x, y = skills[a], skills[b]
        item_probs[i] = max((probs[x]*degree_list[x]+probs[y]*degree_list[y])/(degree_list[x]+degree_list[y]), 0)
    index = torch.multinomial(item_probs, 1, replacement=True)
    a, b = index//length, index%length
    x, y = skills[a], skills[b]
    return [x, y]