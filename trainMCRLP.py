import os
import random
import time
from argparse import ArgumentParser
from random import randint, sample

import numpy as np
import torch
from torch.nn import BCELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from KTModel.DataLoader import DatasetRetrieval, MyDataset2
from Scripts.Envs import KESEnv, KESEnvCo
from Scripts.options import get_options
from Scripts.utlis import load_agent, set_random_seed
from Scripts.Agent.myModel.TGCN.train_utils import create_new_feats, get_initial_logs, attention_encode, \
    get_item_degree_list, get_item_types


def main(args):
    set_random_seed(args.rand_seed)
    if args.model != 'CoKT':
        dataset = MyDataset2(os.path.join(args.data_dir, args.dataset))
        env = KESEnv(dataset, args.model, args.dataset, args.device)
    else:
        dataset = DatasetRetrieval(os.path.join(args.data_dir, args.dataset))
        env = KESEnvCo(dataset, args.worker, args.model, args.dataset, args.device)
    args.skill_num = env.skill_num
    # Create Agent
    model = load_agent(args).to(args.device)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_path = os.path.join(args.save_dir, args.exp_name + '_' + str(args.path))
    if args.load_model:
        model.load_state_dict(torch.load(model_path, args.device))
        print(f"Load Model From {model_path}")
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    criterion = lambda p, r: -torch.mean(r * torch.log(p + 1e-9))
    criterion_kt = BCELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=100, verbose=True, min_lr=args.min_lr)
    print('-' * 20 + "Training Start" + '-' * 20)
    all_mean_rewards, all_rewards = [], []
    skill_num, batch_size = args.skill_num, args.batch_size
    targets, paths = None, None
    best_reward = -1e9
    skill_list, skill_dict, skill_num_dict, feature, degree_list,initial_len = \
        args.skill_list, args.skill_dict, args.skill_num_dict, args.feature, args.degree_list, args.initial_len
    dataset = args.dataset
    feature = feature.to(args.device)
    item_num = feature.shape[0]//2
    degree_list = torch.tensor(degree_list, device=args.device)
    item_degree_list = get_item_degree_list(feature, degree_list)
    skill_index_dict = {}
    for index in range(len(skill_list)):
        skill_index_dict[skill_list[index]] = index
    item_types = get_item_types(initial_len, dataset)
    for epoch in range(args.num_epochs):
        avg_time = 0
        avg_item_time = 0
        model.train()
        epoch_mean_rewards = []
        for i in tqdm(range(50000 // batch_size)):
            t0 = time.time()
            x = []
            batch_targets = []
            batch_logs = []
            batch_initial_logs = []
            for _ in range(batch_size):
                targets = sample(skill_list, 3)
                batch_targets.append(targets)
                targets = torch.tensor(targets, device=args.device).unsqueeze(0)
                # initial_len = 10
                initial_logs = get_initial_logs(skill_num_dict, item_types)
                batch_initial_logs.append(initial_logs)
                # 随机获取10个学习项目
                newfeats = create_new_feats(feature, initial_logs, initial_len)
                # 得到10*123*298的矩阵
                skill_logs = []
                for times in range(initial_len):
                    skill_logs.extend(skill_dict[initial_logs[times]])
                # 获取10个学习项目对应的20个知识概念
                batch_logs.append(skill_logs)
                x.append(newfeats)
            x = torch.stack(x, dim=0)
            batch_targets = torch.tensor(batch_targets, device=args.device)
            batch_logs = torch.tensor(batch_logs, device=args.device)
            time_score = torch.zeros(batch_size, initial_len, feature.shape[1], feature.shape[0], device=args.device)
            for times in range(len(item_types)):
                batch_log = batch_logs[:, :item_types[times]]
                # 分别获取前1，1-2，……，1-10个学习项目对应的知识概念序列
                score, _ = env.begin_episode(batch_targets, batch_log)
                # 放入env，获得对于目标概念的分数
                for batch in range(batch_size):
                    item = batch_initial_logs[batch][times]
                    degree_feature = feature[item] * degree_list
                    degree_feature = degree_feature / item_degree_list[item][0] * item_degree_list[item][1]
                    time_score[batch, times:, :, item + item_num] += degree_feature * score[batch]
                    # 对获得的分数加权处理，加入x矩阵
            x = x + time_score
            x = attention_encode(x, item_num)
            score, _ = env.begin_episode(batch_targets, batch_logs)
            model.begin_episode(batch_targets, x)
            result = model.n_step(args.steps, path_type=args.path)
            paths, pros = result[0], result[1]
            _, scores = env.n_step(paths, binary=True)
            rewards = env.end_episode().unsqueeze(-1)
            loss = criterion(pros, rewards)
            if args.withKT:
                loss += criterion_kt(result[2], scores)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            mean_reward = torch.mean(rewards).item()
            scheduler.step(mean_reward)
            avg_time += time.time() - t0
            epoch_mean_rewards.append(mean_reward)
            all_rewards.append(mean_reward)
            print(f'Epoch:{epoch}\tbatch:{i}\tavg_time:{avg_time / (i + 1):.4f}\t'
                  f'loss:{loss:.4f}\treward:{mean_reward:.4f}')
        print(targets[:10], '\n', paths[:10])
        epoch_mean_rewards = torch.as_tensor(epoch_mean_rewards)
        all_mean_rewards.append(torch.mean(epoch_mean_rewards).item())
        if all_mean_rewards[-1] > best_reward:
            best_reward = all_mean_rewards[-1]
            torch.save(model.state_dict(), model_path)
            print("New Best Result Saved!")
    for i in all_mean_rewards:
        print(i)
    np.save(os.path.join(args.visual_dir, f'{args.exp_name}_{args.path}'), np.array(all_rewards))
    model.eval()
    model.withKt = False
    test_rewards = []
    model.load_state_dict(torch.load(model_path, args.device))
    print(model_path)
    print('-' * 20 + "Testing Start" + '-' * 20)
    with torch.no_grad():
        for i in tqdm(range(50000 // batch_size)):
            x = []
            batch_targets = []
            batch_logs = []
            batch_initial_logs = []
            for _ in range(batch_size):
                targets = sample(skill_list, 3)
                batch_targets.append(targets)
                initial_logs = get_initial_logs(skill_num_dict, item_types)
                batch_initial_logs.append(initial_logs)
                # 随机获取10个学习项目
                newfeats = create_new_feats(feature, initial_logs, initial_len)
                # 得到10*123*298的矩阵
                skill_logs = []
                for times in range(initial_len):
                    skill_logs.extend(skill_dict[initial_logs[times]])
                # 获取10个学习项目对应的20个知识概念
                batch_logs.append(skill_logs)
                x.append(newfeats)
            x = torch.stack(x, dim=0)
            batch_targets = torch.tensor(batch_targets, device=args.device)
            batch_logs = torch.tensor(batch_logs, device=args.device)
            # [[1],[2,3],[4,5],[6,7,8,9,10]]
            # 1,3,5,10
            time_score = torch.zeros(batch_size, initial_len, feature.shape[1], feature.shape[0], device=args.device)
            for times in range(len(item_types)):
                batch_log = batch_logs[:, :item_types[times]]
                # 分别获取前1，1-2，……，1-10个学习项目对应的知识概念序列
                score, _ = env.begin_episode(batch_targets, batch_log)
                # 放入env，获得对于目标概念的分数
                for batch in range(batch_size):
                    item = batch_initial_logs[batch][times]
                    degree_feature = feature[item] * degree_list
                    degree_feature = degree_feature / item_degree_list[item][0] * item_degree_list[item][1]
                    time_score[batch, times:, :, item + item_num] += degree_feature * score[batch]
                    # 对获得的分数加权处理，加入x矩阵
            x = x + time_score
            x = attention_encode(x, item_num)
            score, _ = env.begin_episode(batch_targets, batch_logs)
            model.begin_episode(batch_targets, x)
            result = model.n_step(args.steps, path_type=args.path)
            paths, pros = result[0], result[1]
            env.n_step(paths)
            rewards = env.end_episode().unsqueeze(-1)
            loss = criterion(pros, rewards)
            test_rewards.append(rewards.cpu())
            print(f'batch:{i}\tloss:{loss:.4f}\treward:{torch.mean(rewards):.4f}')
    test_rewards = torch.concat(test_rewards)
    print(paths[:10])
    print(f"Mean Reward for Test:{torch.mean(test_rewards)}")


if __name__ == '__main__':
    torch.set_num_threads(6)
    parser = ArgumentParser("LearningPath-Planing")
    args_ = get_options(parser, {'agent': 'TGCN', 'simulator': 'KES'})
    main(args_)
