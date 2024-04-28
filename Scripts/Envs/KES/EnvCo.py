from multiprocessing import Process, Queue

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence

from .Env import KESEnv


class KESEnvCo(KESEnv):
    def __init__(self, dataset, workers=4, model_name='CoKT', dataset_name='Assist2009', device='cuda'):
        super(KESEnvCo, self).__init__(dataset, model_name, dataset_name, device)
        self.dataset = dataset
        self.his = None

        self.process_list = []
        self.worker_queue = Queue()
        self.index_queue = Queue()
        for i in range(workers):
            p = Process(target=self.worker, args=(self.worker_queue, self.index_queue))  # 实例化进程对象
            p.start()
            self.process_list.append(p)

    def initial_logs(self, logs):
        self.put_index(logs)
        logs = self.get_next(logs)
        return super(KESEnvCo, self).initial_logs(logs)

    def learn_and_test(self, items):
        self.put_index(items)
        logs = self.get_next(items)
        return super(KESEnvCo, self).learn_and_test(logs)

    def exam(self):
        scores = []
        for i in range(self.target_num):
            self.put_index(self.targets[:, i:i + 1], add_to_his=False)
            targets_re = self.get_next(self.targets[:, i:i + 1])
            score, _ = self.model.learn(targets_re, self.states)  # (B,)
            scores.append(score.squeeze(-1))
        return torch.mean(torch.stack(scores, dim=0), dim=0)

    def worker(self, q1: Queue, q2: Queue):
        while True:
            if not q2.empty():
                skill = q2.get()
                q1.put(self.dataset.get_query(-1, skill, range(self.his_len(), len(skill))))

    @staticmethod
    def collate_fn(data_):
        r_his = [_ for l in data_ for _ in l[0]]  # bs*(seq_len*R)个相似学生序列
        r_his = pack_sequence([torch.as_tensor(_, dtype=torch.int32) for _ in r_his], enforce_sorted=False)
        r_skill_y = torch.as_tensor(np.array([_[1] for _ in data_]), dtype=torch.int32)  # (bs,len,R,2) after pad
        return r_his, r_skill_y

    def get_next(self, items):
        batch_size = items.shape[0]
        r = []
        while len(r) < batch_size:
            try:
                x = self.worker_queue.get()
                r.append(x)
            except Exception as e:
                print(e)
                continue
        r = self.collate_fn(r)
        r = [_.to(self.device) for _ in r]
        r.insert(0, items)
        return r

    def put_index(self, index, add_to_his=True):
        index = index.cpu().numpy()
        if not self.his is None:
            index = np.concatenate([self.his, index], axis=1)
        if add_to_his:
            self.his = index
        for i in index:
            self.index_queue.put(i)

    def his_len(self):
        return 0 if self.his is None else self.his.shape[1]

    def __del__(self):
        for p in self.process_list:
            p.terminate()

    def end_episode(self, *args, **kwargs):
        self.his = None
        return super(KESEnvCo, self).end_episode()
