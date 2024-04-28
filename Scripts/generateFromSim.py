import argparse
import os
import sys

import numpy as np
import torch
from torch import device
from tqdm import tqdm

sys.path.append('../')
from KTModel.DataLoader import MyDataset2, DatasetRetrieval
from Scripts.Envs.KES.Env import KESEnv
from Scripts.Envs.KES.EnvCo import KESEnvCo


def generateData(args):
    if args.model != 'CoKT':
        dataset = MyDataset2(os.path.join(args.data_dir, args.dataset))
        env = KESEnv(dataset, args.model, args.dataset, args.device)
    else:
        dataset = DatasetRetrieval(os.path.join(args.data_dir, args.dataset))
        env = KESEnvCo(dataset, 6, args.model, args.dataset, args.device)
    skills = np.random.randint(env.skill_num, size=(args.size, args.real_len))
    real_len = np.zeros(args.size, dtype=np.int32) + args.real_len
    targetsPlace = torch.zeros((args.batch_size, 1), device=args.device)
    ys = []
    for i in tqdm(range(0, args.size, args.batch_size)):
        skillsT = torch.as_tensor(skills[i: min(i + args.batch_size, args.size)], device=args.device)
        env.begin_episode(targetsPlace[:min(args.batch_size, args.size - i)])

        _, y = env.n_step(skillsT, binary=True)
        ys.append(y.tolist())
        env.end_episode()

    ys = np.concatenate(ys).astype(bool).squeeze(-1)
    result = {'skill': skills, 'real_len': real_len, 'y': ys}

    targetFolder = os.path.join(args.data_dir, args.dataset + args.model)
    os.makedirs(targetFolder, exist_ok=True)
    np.savez(os.path.join(targetFolder, args.dataset + args.model + '.npz'), **result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--data_dir', type=str, default='../data/')
    # options
    parser.add_argument('-d', '--dataset', type=str, default='assist09', choices=['assist09', 'junyi', 'assist15'])
    parser.add_argument('-m', '--model', type=str, default='DKT', choices=['DKT', 'CoKT'])
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('--size', type=int, default=50000)
    parser.add_argument('--real_len', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)

    args_ = parser.parse_args()

    device_name = 'cpu' if args_.cuda < 0 else 'cuda:{}'.format(args_.cuda)
    args_.device = device(device_name)
    generateData(args_)
