# coding: utf-8
# 2021/1/28 @ tongshiwei
from pprint import pformat
from typing import Sequence
from gym.spaces import Space

__all__ = ["ListSpace"]  # 表示import * 只取目标类


class ListSpace(Space):
    def __init__(self, elements: Sequence, seed=None):
        super(ListSpace, self).__init__(shape=(len(elements),))
        self.elements = elements
        self.seed(seed)

    def sample(self, mask=None):
        return self.np_random.choice(self.elements)

    def sample_idx(self):
        return self.np_random.choice(range(len(self.elements)))

    def sample_path(self, n):
        return self.np_random.choice(self.elements, n, replace=True)

    def sample_paths(self, k, n):
        return self.np_random.choice(self.elements, k * n, replace=True).reshape((k, n))

    def contains(self, item):
        return item in self.elements

    def __repr__(self):
        return pformat(self.elements)

    def __getitem__(self, item):
        return self.elements[item]

    @property
    def is_np_flattenable(self):
        raise NotImplementedError
