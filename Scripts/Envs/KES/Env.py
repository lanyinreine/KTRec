import torch

from .utils import load_d_agent
from ..meta import Env
from ..shared.KSS_KES import episode_reward
from ..spaces import ListSpace


class KESEnv(Env):
    def __init__(self, dataset, model_name='DKT', dataset_name='Assist2009', device='cuda'):
        self.skill_num = dataset.feats_num
        self.action_space = ListSpace(range(self.skill_num))
        self.model = load_d_agent(model_name, dataset_name, self.skill_num, device)
        self.device = device
        self.states = None
        self.targets = None
        self.initial_score = None
        self.target_num = None

    @property
    def parameters(self) -> dict:
        return {
            "action_space": self.action_space
        }

    def initial_logs(self, logs):
        scores, self.states = self.model.learn(logs)
        # states为长度为2的元组，内部是两个1*256*128的张量
        return scores.squeeze(-1)

    def learn_and_test(self, items):
        scores, self.states = self.model.learn(items, self.states)
        return items, scores

    def exam(self):
        scores = []
        for i in range(self.target_num):
            score, _ = self.model.learn(self.targets[:, i:i + 1], self.states)  # (B,)
            scores.append(score.squeeze(-1))
        return torch.mean(torch.stack(scores, dim=0), dim=0)

    def separate_exam(self, targets, initial_logs=None):
        self.targets = targets
        self.target_num = targets.shape[-1]
        _, _ = self.initial_logs(initial_logs)
        scores = []
        for i in range(self.target_num):
            score, _ = self.model.learn(self.targets[:, i:i + 1], self.states)  # (B,)
            scores.append(score.squeeze(-1))
        return torch.stack(scores, dim=0)

    def begin_episode(self, targets, initial_logs=None):
        self.targets = targets
        self.target_num = targets.shape[-1]
        if initial_logs is not None:
            score = self.initial_logs(initial_logs)
            # score=batch_size*initial_len
            initial_logs = torch.stack([initial_logs.float(), score], dim=-1)
            # initial_logs=batch_size*initial_len*2
        self.initial_score = self.exam()
        return self.initial_score, initial_logs

    def end_episode(self, *args, **kwargs):
        final_score = self.exam()
        reward = episode_reward(self.initial_score, final_score, 1).squeeze()
        self.states = None
        if 'score' in kwargs:
            return final_score, reward
        return reward

    def step(self, learning_item_id: torch.Tensor, binary=False, *args, **kwargs):
        return self.n_step(learning_item_id.unsqueeze(1), binary)

    def n_step(self, learning_path, binary=False, *args, **kwargs):
        items, scores = self.learn_and_test(learning_path)
        if binary:
            # scores = torch.bernoulli(scores)
            scores = torch.where(scores > 0.5, torch.ones_like(scores), torch.zeros_like(scores))
            # According to the students' mastering degree, the right and wrong are given randomly,
            # to avoid the KT model directly learning the optimal solution
        return items, scores

    def reset(self):
        self.targets, self.states = None, None

    def create_data(self, batch_size, seq_len):
        learning_path = torch.randint(self.skill_num, (batch_size, seq_len), device=self.device)
        _, score = self.learn_and_test(learning_path)
        return learning_path, score.squeeze(-1)
