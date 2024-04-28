import torch
from torch.utils.data import Dataset

from Scripts.Envs.KSS.EnvMulti import KSSEnvMulti


class KSSDataset(Dataset):
    def __init__(self, env: KSSEnvMulti, size):
        super().__init__()
        self.env = env
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        learner, initial_score = self.env.begin_episode()
        return learner, (learner.profile['target'], learner.profile['logs'], initial_score)

    @staticmethod
    def collate_fn(data):
        learners, data = list(zip(*data))
        data = list(zip(*data))
        data = [torch.as_tensor(_) for _ in data]
        return learners, data


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    env_ = KSSEnvMulti(targets_num=3, workers=6, initial_step=10)
    dataset = KSSDataset(env_, 50000)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=6, prefetch_factor=True,
                            collate_fn=dataset.collate_fn)
    for i, x in enumerate(tqdm(dataloader)):
        learners_, data_ = x
        targets, initial_logs, initial_scores = [_.to('cuda') for _ in data_]
        learning_paths = torch.randint(env_.skills_num, (targets.size(0), 20))
        _, scores = env_.n_step(learners_, learning_paths)
        rewards = env_.end_episode(learners_, initial_scores)
        print(torch.mean(rewards))
    env_.delete()
