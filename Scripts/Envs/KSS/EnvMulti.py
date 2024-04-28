from multiprocessing import Pool

import numpy as np
import torch

from .Env import KSSEnv
from ..shared.KSS_KES import episode_reward


class KSSEnvMulti(KSSEnv):
    def __init__(self, targets_num=3, workers=4, seed=None, initial_step=20):
        super(KSSEnvMulti, self).__init__(seed, initial_step)
        self.targets_num = targets_num
        self.initial_step = initial_step
        self.skills_num = self.action_space.shape[0]
        self.pool = Pool(workers)

    def begin_episode(self, *args, **kwargs):
        target = np.random.choice(self.skills_num, self.targets_num, replace=False).tolist()
        learner = self.learners.one_generate(target)
        logs = list(np.random.randint(self.skills_num, size=(self.initial_step,)))
        scores = super(KSSEnvMulti, self).n_step(learner, logs)
        learner.update_logs(list(zip(logs, scores)))
        # self._initial_logs(learner)
        initial_score = self._exam(learner)
        return learner, initial_score

    def n_step(self, learners, learning_paths: torch.Tensor, *args, **kwargs):
        learners, scores = zip(*self.pool.starmap(self.w1, zip(learners, learning_paths.tolist())))
        device = learning_paths.device
        scores = torch.as_tensor(scores, device=device)
        return learners, learning_paths, scores

    def end_episode(self, learners, initial_scores, *args, **kwargs):
        final_scores = self.pool.map(self.w2, learners)
        device = initial_scores.device
        final_scores = torch.as_tensor(final_scores, device=device)
        reward = episode_reward(initial_scores, final_scores, 1.0).squeeze()
        if 'score' in kwargs:
            return final_scores, reward
        return reward

    def w1(self, learner, path):
        score = super(KSSEnvMulti, self).n_step(learner, path)
        return learner, score

    def w2(self, learner):
        return self._exam(learner)

    def delete(self):
        self.pool.close()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']

        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
