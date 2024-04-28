import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import BCELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from KTModel.DataLoader import MyDataset2, DatasetRetrieval
from Scripts.Envs.KES.Env import KESEnv
from Scripts.Envs.KES.EnvCo import KESEnvCo
from Scripts.options import get_options
from Scripts.utlis import load_agent
from Scripts.utlis import set_random_seed


def main(args):
    print(args)
    set_random_seed(args.rand_seed)
    # Create Dataset
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
    criterion = BCELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=100, verbose=True, min_lr=args.min_lr)
    print('-' * 20 + "Training Start" + '-' * 20)
    all_mean_rewards, all_rewards = [], []
    skill_num, batch_size = args.skill_num, args.batch_size
    targets, paths = None, None
    best_reward = -1e9
    initial_len = args.initial_len
    skill_len = initial_len*3
    for epoch in range(args.num_epochs):
        avg_time = 0
        model.train()
        epoch_mean_rewards = []
        for i in tqdm(range(50000 // batch_size)):
            t0 = time.time()
            targets, initial_logs = torch.randint(skill_num, (batch_size, 3), device=args.device), \
                                    torch.randint(skill_num, (batch_size, skill_len), device=args.device)
            initial_scores, initial_logs = env.begin_episode(targets, initial_logs)

            model.begin_episode(targets, initial_logs)
            initial_pres = model.exam()
            paths, his_pres = model.n_step(args.steps, path_type=args.path)
            final_pres = model.exam()
            pres = torch.cat([initial_pres, his_pres.view(-1), final_pres])
            _, his_scores = env.n_step(paths)
            final_scores, rewards = env.end_episode(score=True)
            scores = torch.cat([initial_scores.view(-1), his_scores.view(-1), final_scores.view(-1)])
            scores = torch.where(torch.greater(scores, 0.5), torch.ones_like(scores), torch.zeros_like(scores))

            loss = criterion(pres, scores)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            scheduler.step(loss)
            avg_time += time.time() - t0
            mean_reward = torch.mean(rewards).item()
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
    test_rewards = []
    model.load_state_dict(torch.load(model_path, args.device))
    with torch.no_grad():
        for i in tqdm(range(50000 // batch_size)):
            targets, initial_logs = torch.randint(skill_num, (batch_size, 3), device=args.device), \
                                    torch.randint(skill_num, (batch_size, skill_len), device=args.device)
            initial_scores, initial_logs = env.begin_episode(targets, initial_logs)
            model.begin_episode(targets, initial_logs)
            initial_pres = model.exam()
            paths, his_pres = model.n_step(args.steps, path_type=args.path)
            final_pres = model.exam()
            pres = torch.cat([initial_pres, his_pres.view(-1), final_pres])
            _, his_scores = env.n_step(paths)
            final_scores, rewards = env.end_episode(score=True)
            scores = torch.cat([initial_scores.view(-1), his_scores.view(-1), final_scores.view(-1)])
            scores = torch.where(torch.greater(scores, 0.5), torch.ones_like(scores), torch.zeros_like(scores))

            loss = criterion(pres, scores)
            test_rewards.append(rewards.cpu())
            print(f'batch:{i}\tloss:{loss:.4f}\treward:{torch.mean(rewards):.4f}')
    test_rewards = torch.concat(test_rewards)
    print(paths[:10])
    print(f"Mean Reward for Test:{torch.mean(test_rewards)}")


if __name__ == '__main__':
    torch.set_num_threads(6)
    parser = ArgumentParser("LearningPath-Planing")
    args_ = get_options(parser, {'agent': 'MPC', 'simulator': 'KES'})
    main(args_)
