from torch import rand, argsort, randint, randint_like
from random import sample


def generate_path(bs, skill_num, path_type, n, device):
    if path_type == 0 or path_type == 1:
        origin_path = argsort(rand((bs, n), device=device), dim=-1)  # 1-N知识点
        if path_type == 1:  # 所有知识点按N大小分组
            origin_path += n * randint_like(origin_path[:, 0:1], skill_num // n)
    else:  # 2 or 3
        origin_path = argsort(rand((bs, skill_num), device=device), dim=-1)  # 所有知识点排序取topN
        if path_type == 2:
            origin_path = origin_path[:, :n]  # 随机取N个知识点排序
    return origin_path
