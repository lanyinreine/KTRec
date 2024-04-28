import os

import numpy as np
from elasticsearch import Elasticsearch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_folder):
        super(MyDataset, self).__init__()
        folder_name = os.path.basename(data_folder)
        with np.load(os.path.join(data_folder, folder_name + '.npz')) as data:
            self.y, self.skill, self.problem, self.real_len = \
                data['y'], data['skill'], data['problem'], data['real_len']
        zero_problem = (self.problem == 0)
        self.problem += np.max(self.skill)
        self.problem[zero_problem] = 0
        self.user_nums = len(self.real_len)
        self.fea_nums = np.max(self.problem) + 1
        self.mode_len = {'train': 0.6, 'valid': 0.8, 'test': 1}  # 需要使用的序列长度比例
        self.mode_assist_len = {'train': 0, 'valid': 0.6, 'test': 0.8}  # 不参与label预测的比例
        self.mode = 'train'

    def __len__(self):
        return len(self.real_len)

    def change_mode(self, mode):
        self.mode = mode

    def __getitem__(self, item):
        real_len = int(self.real_len[item] * self.mode_len[self.mode])
        return item, np.stack(
            [self.skill[item][:real_len], self.problem[item][:real_len], self.y[item][:real_len]],
            axis=-1)


class MyDataset2(Dataset):
    def __init__(self, data_folder):
        super(MyDataset2, self).__init__()
        folder_name = os.path.basename(data_folder)
        self.dataset_name = folder_name
        with np.load(os.path.join(data_folder, folder_name + '.npz'), allow_pickle=True) as data:
            y, skill, real_len = data['y'], data['skill'], data['real_len']
            if folder_name == 'junyi':
                skill = data['problem'] - 1
        train_data, test_data = self.train_test_split([y, skill, real_len])
        self.data = {'train': train_data, 'valid': test_data, 'test': test_data}
        self.mode = 'train'
        try:
            self.feats_num = np.max(skill) + 1
        except ValueError:
            self.feats_num = np.max(np.concatenate(skill)) + 1
        self.users_num = len(real_len)

    @staticmethod
    def train_test_split(data, split=0.8):
        n_samples = data[0].shape[0]
        users = list(range(len(data[0])))
        split_point = int(n_samples * split)
        train_data, test_data = [users[:split_point], ], [users[split_point:]]
        for d in data:
            train_data.append(d[:split_point])
            test_data.append(d[split_point:])
        return train_data, test_data

    def __len__(self):
        return len(self.data[self.mode][0])

    def change_mode(self, mode):
        self.mode = mode

    def __getitem__(self, item):
        user, y, skill, real_len = [_[item] for _ in self.data[self.mode]]
        return user, np.stack([skill[:real_len], y[:real_len]], axis=-1)


class DatasetRec(MyDataset2):
    # For GRU4Rec, Predict the next item
    def __getitem__(self, item):
        user, y, skill, real_len = [_[item] for _ in self.data[self.mode]]
        return user, np.stack([skill[:real_len - 1], skill[1:real_len]], axis=-1)


class DatasetRetrieval(MyDataset2):
    def __init__(self, data_folder, r=5):
        super(DatasetRetrieval, self).__init__(data_folder)
        self.es = Elasticsearch(hosts=['http://localhost:9200/']).options(
            request_timeout=20,
            retry_on_timeout=True,
            ignore_status=[400, 404]
        )
        self.safe_query = self.get_safe_query()
        self.R = r

    def get_safe_query(self):
        _, ys, skills, _ = self.data['train']
        print("skills.type", skills.dtype)
        print("ys.type", ys.dtype)
        safe_query = np.stack([skills[:, 0], ys[:, 0].astype(np.int32)], -1)
        # safe_query = np.stack([skills, ys.astype(np.int32)], -1)
        return safe_query

    def get_query(self, user, skills, index_range):
        safe_user = np.random.choice(self.data['train'][0], self.R + 1, replace=False)
        safe_user = safe_user[safe_user != user][:self.R]
        safe_query = self.safe_query[safe_user]
        queries = []
        skills = skills.astype('str')
        skills_str = ' '
        index = f'{self.dataset_name}_train'
        must_not_query = {'bool': {'must_not': {'term': {'user': user}}}}
        for _ in index_range:
            skill = skills[_]
            skills_str += ' ' + skill
            query = [{'index': index},
                     {'size': self.R,
                      'query': {'bool': {
                          'filter':
                              [{'term': {'skill': skill}}, must_not_query],
                          'must': {'match': {'history': skills_str}}}},
                      '_source': ['history', 'y']}]
            queries.extend(query)
        result = self.es.msearch(index=index, searches=queries)['responses']
        r_his, r_skill_y = [], []
        for rs in result:  # seq_len个
            skill_y = []
            rs = rs['hits']['hits']
            for r in rs:  # R个
                r = r['_source']
                his = np.array(r['history'].split(' ')).astype(int)
                his = np.stack([his, np.array(r['y'], dtype=int)], axis=-1)
                if his.ndim == 1:
                    his = np.expand_dims(his, 0)
                r_his.append(his)
                skill_y.append(his[-1])
            for _ in range(self.R - len(rs)):
                r_his.append(safe_query[_:_ + 1])
                skill_y.append(safe_query[_])
            r_skill_y.append(np.stack(skill_y, axis=0))  # (R, 2)
        return r_his, np.stack(r_skill_y, axis=0)

    def __getitem__(self, item):
        user, y, skill, real_len = [_[item] for _ in self.data[self.mode]]
        r_his, r_skill_y = self.get_query(user, skill, range(real_len))
        return user, np.stack([skill[:real_len], y[:real_len]], axis=-1), r_his, r_skill_y


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from utils import collate_co, collate_fn
    from torch import nn
    from BackModels import CoKT

    dataset = DatasetRetrieval('../data/assist09')
    dataloader = DataLoader(dataset, 128, True, num_workers=4, collate_fn=collate_co)
    i = 0
    dataset.change_mode('train')
    e = nn.Embedding(200, 4)
    # rnn = nn.LSTM(4, 9, batch_first=True)
    rnn = CoKT(4, 16, 0.4)
    for users_, logs_packed_, r_his_, r_skill_y_, y_ in dataloader:
        print(users_)
        print(logs_packed_.data.shape)
        # print(r_his.data.shape)
        # print(r_skill_y.data.shape)
        # logs_packed = PackedSequence(e(logs_packed.data[:, 0]), logs_packed.batch_sizes)
        # r_his = PackedSequence(e(r_his.data[:, 0]), r_his.batch_sizes)
        # r_skill_y = PackedSequence(e(r_skill_y.data[:, :, 0]), r_skill_y.batch_sizes)
        # o = rnn(logs_packed, r_his, r_skill_y)
        # print(o, o.shape)
        # i += 1
        # if i >= 1:
        #     break
