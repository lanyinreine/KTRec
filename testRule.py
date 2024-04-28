import argparse
import os

import torch
from longling import path_append, abs_current_dir
from tqdm import tqdm

from KTModel.Configure import get_exp_configure
from KTModel.DataLoader import MyDataset2, DatasetRetrieval
from KTModel.utils import load_model
from Scripts.Envs import KESEnv
from Scripts.Envs.KES.EnvCo import KESEnvCo


def rank(env, path, device):
    path = path.reshape(-1, 1)
    targets = torch.randint(env.skill_num, size=(path.size(0), 3), device=device)
    initial_logs = torch.randint(env.skill_num, size=(path.size(0), 10), device=device)

    env.begin_episode(targets, initial_logs)
    env.n_step(path)
    rewards = env.end_episode()
    return path.reshape(-1)[torch.argsort(rewards)]


def generate_path(bs, path_type, n, skill_num, device):
    if path_type == 0 or path_type == 1:
        origin_path = torch.argsort(torch.rand((bs, n), device=device), dim=-1)  # 1-20的知识点
        if path_type == 1:
            origin_path += n * torch.randint_like(origin_path[:, 0:1], skill_num // n)
            # 所有知识点按20为大小分组
    else:  # 2 or 3
        origin_path = torch.argsort(torch.rand((bs, skill_num), device=device), dim=-1)  # 所有知识点排序取top20
        if path_type == 2:
            origin_path = origin_path[:, :n]  # 随机取20个知识点排序
    return origin_path


def testRuleBased(args):
    initial_len = args.initial_len
    skill_len = initial_len*3
    if args.model != 'CoKT':
        dataset = MyDataset2(os.path.join(args.data_dir, args.dataset))
        env = KESEnv(dataset, args.model, args.dataset, args.device)
    else:
        dataset = DatasetRetrieval(os.path.join(args.data_dir, args.dataset))
        env = KESEnvCo(dataset, 6, args.model, args.dataset, args.device)
    origin_paths = generate_path(args.batch, args.p, args.n, env.skill_num, args.device)
    targets = torch.randint(env.skill_num, size=(args.batch, 3), device=args.device)
    initial_logs = torch.randint(env.skill_num, size=(args.batch, skill_len), device=args.device)
    if args.agent == 'rule':
        ranked_paths = torch.stack([rank(env, origin_paths[i], args.device)[-args.n:] for i in tqdm(range(args.batch))],
                                   dim=0)
    elif args.agent == 'random':
        ranked_paths = origin_paths[:, -args.n:]
    elif args.agent == 'GRU4Rec':
        d_model = load_d_agent(args.agent, args.dataset, env.skill_num, args.device, False)
        a1 = torch.arange(args.batch).unsqueeze(-1).repeat_interleave(dim=1, repeats=origin_paths.size(-1))
        selected_paths = torch.ones((args.batch, env.skill_num), dtype=torch.bool, device=args.device)
        selected_paths[a1, origin_paths] = False
        a1 = a1[:, 0]
        path, states = initial_logs, None
        ranked_paths = []
        for _ in tqdm(range(args.n)):
            o, states = d_model.learn(path, states)
            o = o[:, -1]
            o[selected_paths] = -1
            path = torch.argmax(o, dim=-1, keepdim=True)
            ranked_paths.append(path)
            selected_paths[a1, path.squeeze(-1)] = True
        ranked_paths = torch.cat(ranked_paths, dim=-1)
    else:
        raise NotImplementedError
    print(ranked_paths[:10])
    env.begin_episode(targets, initial_logs)
    env.n_step(ranked_paths)
    print(torch.mean(env.end_episode()))


def load_d_agent(model_name, dataset_name, skill_num, device, with_label=True):
    model_parameters = get_exp_configure({'model': model_name, 'dataset': dataset_name})
    model_parameters.update(
        {'feat_nums': skill_num, 'model': model_name, 'without_label': not with_label, 'output_size': skill_num})
    model = load_model(model_parameters).to(device)
    model_folder = path_append(abs_current_dir(__file__), os.path.join('SavedModels'))
    model_path = os.path.join(model_folder, model_name + '_' + dataset_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--data_dir', type=str, default='../data')
    # options
    parser.add_argument('-d', '--dataset', type=str, default='ednet', choices=['assist09', 'junyi', 'assist15', 'ednet'])
    parser.add_argument('-a', '--agent', type=str, default='random', choices=['rule', 'random', 'GRU4Rec'])
    parser.add_argument('-m', '--model', type=str, default='DKT', choices=['DKT', 'CoKT'])
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('-b', '--batch', type=int, default=128)
    parser.add_argument('-n', type=int, default=10)
    parser.add_argument('-p', type=int, default=0)
    parser.add_argument('--seed', type=int, default=16)
    parser.add_argument('--initial_len', type=int, default=10)

    args_ = parser.parse_args()

    device_name = 'cpu' if args_.cuda < 0 else f'cuda:{args_.cuda}'
    args_.device = torch.device(device_name)
    testRuleBased(args_)
