import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from KTModel.DataLoader import MyDataset2, DatasetRetrieval
from Scripts.Envs import KESEnv
from Scripts.Envs.KES.EnvCo import KESEnvCo
from Scripts.Envs.KES.utils import load_d_agent
from Scripts.options import get_options
from Scripts.utlis import load_agent, set_random_seed, ReplayBuffer


def main(args):
    print(args)
    set_random_seed(args.rand_seed)
    if args.model != 'CoKT':
        dataset = MyDataset2(os.path.join(args.data_dir, args.dataset))
        env = KESEnv(dataset, args.model, args.dataset, args.device)
    else:
        print(os.path.join(args.data_dir, args.dataset))
        dataset = DatasetRetrieval(os.path.join(args.data_dir, args.dataset))
        env = KESEnvCo(dataset, args.worker, args.model, args.dataset, args.device)
    args.skill_num = env.skill_num
    # Create Agent
    model = load_agent(args).to(args.device)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    d_model = load_d_agent('DKT', args.dataset+'DKT', env.skill_num, args.device, False)
    model.get_d_model(d_model)
    model_path = os.path.join(args.save_dir, args.exp_name + str(args.path))
    if args.load_model:
        model.load_state_dict(torch.load(model_path, args.device))
        print(f"Load Model From {model_path}")
    replay_buffer = ReplayBuffer(50)
    print('-' * 20 + "Training Start" + '-' * 20)
    all_mean_rewards, all_rewards = [], []
    skill_num, batch_size, initial_len = args.skill_num, args.batch_size, args.initial_len
    skill_len = initial_len*3
    targets, paths = None, None
    best_reward = -1e9
    for epoch in range(args.num_epochs):
        avg_time = 0
        model.train()
        epoch_mean_rewards = []
        for i in tqdm(range(50000 // batch_size)):
            t0 = time.time()
            targets, initial_logs = torch.randint(skill_num, (batch_size, 3), device=args.device), \
                                    torch.randint(skill_num, (batch_size, skill_len), device=args.device)
            env.begin_episode(targets, initial_logs)
            h, hc = model.begin_episode(targets, initial_logs)
            paths, hs = model.n_steps(h, args.steps, path_type=args.path, step_by_step=True)
            env.n_step(paths)
            rewards = env.end_episode()
            mean_reward = torch.mean(rewards).item()
            hs = torch.cat([h.unsqueeze(1), hs], dim=1)
            for j in range(args.steps):
                replay_buffer.add(hs[:, j].cpu().detach().numpy(), paths[:, j].cpu().numpy(), rewards.cpu().numpy(),
                                  hs[:, j + 1].cpu().detach().numpy(), False)
                if replay_buffer.size() > 20:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(1)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    model.update(transition_dict)
            avg_time += time.time() - t0
            epoch_mean_rewards.append(mean_reward)
            all_rewards.append(mean_reward)
            print(f'Epoch:{epoch}\tbatch:{i}\tavg_time:{avg_time / (i + 1):.4f}\treward:{mean_reward:.4f}')
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
            targets, initial_logs = torch.randint(skill_num, (batch_size, 3)).to(args.device), \
                                    torch.randint(skill_num, (batch_size, skill_len)).to(args.device)
            env.begin_episode(targets, initial_logs)
            h, hc = model.begin_episode(targets, initial_logs)
            paths, hs = model.n_steps(h, args.steps, path_type=args.path, step_by_step=True)
            env.n_step(paths)
            rewards = env.end_episode().unsqueeze(-1)
            test_rewards.append(rewards.cpu())
            print(f'batch:{i}\treward:{torch.mean(rewards):.4f}')
    test_rewards = torch.concat(test_rewards)
    print(paths[:10])
    print(f"Mean Reward for Test:{torch.mean(test_rewards)}")


if __name__ == '__main__':
    torch.set_num_threads(6)
    parser = ArgumentParser("LearningPath-Planing")
    args_ = get_options(parser, {'agent': 'DQN', 'simulator': 'KES'})
    main(args_)
