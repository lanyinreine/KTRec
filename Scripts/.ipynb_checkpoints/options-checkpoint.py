import csv
from argparse import ArgumentParser, Namespace

import numpy as np
import torch


def get_exp_configure(agent):
    config_dict = {
        'embed_size': 48,
        'hidden_size': 64,
        'output_size': 1,
        'dropout': 0.5,
        'decay_step': 1000,
        'min_lr': 1e-5,
        'l2_reg': 1e-4,
        'predict_hidden_sizes': [256, 64, 16]
    }
    if agent == 'RR':
        feature_path = '../data/ednet/matrix/feature.csv'
        with open(feature_path, 'r') as file:
            csv_reader = csv.reader(file)
            rows = [row for row in csv_reader]
        feature = [row[1:] for row in rows[1:]]
        feature = [[int(item) for item in row] for row in feature]
        feature = torch.tensor(feature, dtype=torch.int32)
        config_dict['feature'] = feature
        skill_list = rows[0][1:]
        skill_list = [int(x) for x in skill_list]
        config_dict['skill_list'] = skill_list
        skill_dict = {}
        skill_num_list = []
        skill_num_dict = {}
        for i in range(int(len(feature) / 2)):
            skill = []
            for j in range(len(feature[i])):
                if feature[i][j] == 1:
                    skill.append(skill_list[j])
            skill_dict[i] = skill
            length = len(skill)
            if length not in skill_num_list:
                skill_num_dict[length] = [i]
                skill_num_list.append(length)
            else:
                skill_num_dict[length].append(i)
        config_dict['skill_dict'] = skill_dict
        config_dict['skill_num_dict'] = skill_num_dict
    if agent.startswith('MPC'):
        config_dict.update({'hor': 20})
    if agent == 'DQN':
        config_dict['hidden_size'] = 128
        feature_path = '../data/ednet/matrix/feature.csv'
        with open(feature_path, 'r') as file:
            csv_reader = csv.reader(file)
            rows = [row for row in csv_reader]
        feature = [row[1:] for row in rows[1:]]
        feature = [[int(item) for item in row] for row in feature]
        feature = torch.tensor(feature, dtype=torch.int32)
        config_dict['feature'] = feature
        skill_list = rows[0][1:]
        skill_list = [int(x) for x in skill_list]
        config_dict['skill_list'] = skill_list
        skill_dict = {}
        skill_num_list = []
        skill_num_dict = {}
        for i in range(int(len(feature) / 2)):
            skill = []
            for j in range(len(feature[i])):
                if feature[i][j] == 1:
                    skill.append(skill_list[j])
            skill_dict[i] = skill
            length = len(skill)
            if length not in skill_num_list:
                skill_num_dict[length] = [i]
                skill_num_list.append(length)
            else:
                skill_num_dict[length].append(i)
        config_dict['skill_dict'] = skill_dict
        config_dict['skill_num_dict'] = skill_num_dict
    if agent == 'TGCN':
        config_dict['embed_size'] = 123
        adj_path = '../data/ednet/matrix/adj_m.csv'
        with open(adj_path, 'r') as file:
            csv_reader = csv.reader(file)
            rows = [row for row in csv_reader]
        adj = np.array([row[1:] for row in rows[1:]], dtype=int)
        config_dict['adj'] = adj
        degree_list = []
        for i in range(len(adj)):
            degree_list.append(np.sum(adj[i]))
        config_dict['degree_list'] = degree_list
        feature_path = '../data/ednet/matrix/feature.csv'
        with open(feature_path, 'r') as file:
            csv_reader = csv.reader(file)
            rows = [row for row in csv_reader]
        feature = [row[1:] for row in rows[1:]]
        feature = [[int(item) for item in row] for row in feature]
        feature = torch.tensor(feature, dtype=torch.int32)
        config_dict['feature'] = feature
        skill_list = rows[0][1:]
        skill_list = [int(x) for x in skill_list]
        config_dict['skill_list'] = skill_list
        skill_dict = {}
        skill_num_list = []
        skill_num_dict = {}
        for i in range(int(len(feature)/2)):
            skill = []
            for j in range(len(feature[i])):
                if feature[i][j] == 1:
                    skill.append(skill_list[j])
            skill_dict[i] = skill
            length = len(skill)
            if length not in skill_num_list:
                skill_num_dict[length] = [i]
                skill_num_list.append(length)
            else:
                skill_num_dict[length].append(i)
        config_dict['skill_dict'] = skill_dict
        config_dict['skill_num_dict'] = skill_num_dict
    return config_dict


def get_options(parser: ArgumentParser, reset_args=None):
    from torch import device
    if reset_args is None:
        reset_args = {}
    simulator = ['KSS', 'KES']
    agent = ['MPC', 'DQN', 'RR', 'TGCN']
    model = ['DKT', 'CoKT']
    dataset = ['junyi', 'assist09', 'assist09KES', 'assist15', 'ednet']
    parser.add_argument('-s', '--simulator', type=str, choices=simulator, default='KSS')
    parser.add_argument('-a', '--agent', type=str, choices=agent, default='RR')
    parser.add_argument('-m', '--model', type=str, choices=model, default='DKT', help='Model used in MPC or KES')
    parser.add_argument('-d', '--dataset', type=str, choices=dataset, default='ednet')
    parser.add_argument('-w', '--worker', type=int, default=6)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-p', '--path', type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--save_dir', type=str, default='./SavedModels')
    parser.add_argument('--visual_dir', type=str, default='./VisualResults')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--withKT', action='store_true', default=True, help='Whether to use KT as a secondary task')
    parser.add_argument('--binary', action='store_true', default=False, help='Whether the reward is binary')
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--rand_seed', type=int, default=16)
    parser.add_argument('--initial_len', type=int, default=10)
    parser.set_defaults(**reset_args)
    args = parser.parse_args()
    # Get experiment configuration
    exp_configure = get_exp_configure(args.agent)
    args = Namespace(**vars(args), **exp_configure)

    if args.simulator == 'KES':
        args.simulator += args.model
    args.exp_name = '_'.join([args.agent, args.simulator, args.dataset])
    if args.postfix != '':
        args.exp_name += '_' + args.postfix

    device_name = 'cpu' if args.cuda < 0 else f'cuda:{args.cuda}'
    args.device = device(device_name)
    return args
