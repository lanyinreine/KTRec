import argparse
import os
import sys

import torch
from torch import device

sys.path.append('../')
from Scripts.Envs.KES.EnvCo import KESEnvCo
from KTModel.DataLoader import DatasetRetrieval


def testEnv(args):
    # torch.manual_seed(args.seed)
    dataset = DatasetRetrieval(os.path.join(args.data_dir, args.dataset))
    env = KESEnvCo(dataset, args.worker, args.model, args.dataset, args.device)
    targets = torch.randint(dataset.feats_num, (dataset.feats_num, 3), device=args.device)
    initial_logs = torch.randint(dataset.feats_num, (dataset.feats_num, 10), device=args.device)
    # initial_logs=None
    env.begin_episode(targets, initial_logs)
    learning_items = torch.rand((targets.size(0), env.skill_num), device=args.device)
    learning_items = torch.argsort(learning_items, dim=-1)[:, :20]
    # learning_items = torch.rand((dataset.fea_nums, 20), device=args.device)
    # learning_items = torch.argsort(learning_items, dim=-1)
    # learning_items += 20 * torch.randint(env.skill_num // 20, size=(targets.size(0), 1),
    #                                      device=args.device)
    print(learning_items[-1])
    env.n_step(learning_items)
    reward = env.end_episode()
    print(torch.sort(reward))
    print(torch.mean(reward))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--data_dir', type=str, default='../data/')
    # options
    parser.add_argument('--dataset', type=str, default='assist09')
    parser.add_argument('-m', '--model', type=str, default='CoKT')
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('-w', '--worker', type=int, default=6)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    device_name = 'cpu' if args.cuda < 0 else 'cuda:{}'.format(args.cuda)
    args.device = device(device_name)
    testEnv(args)
