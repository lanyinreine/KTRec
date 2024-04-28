# coding: utf-8
# 2021/2/18 @ tongshiwei

import os

from longling import json_load, path_append, abs_current_dir

from ..shared.KSS_KES import KS
from ..utils.io_lib import load_ks_from_csv

"""
Example
-------
>>> load_item("filepath_that_do_not_exsit/not_exsit77250")
{}
"""


def load_items(filepath):
    if os.path.exists(filepath):
        s = json_load(filepath)
        s = {int(_): s[_] for _ in s.keys()}
        return s
    else:
        return {}


def load_knowledge_structure(filepath):
    knowledge_structure = KS()
    knowledge_structure.add_edges_from([list(map(int, edges)) for edges in load_ks_from_csv(filepath)])
    return knowledge_structure


def load_learning_order(filepath):
    return json_load(filepath)


def load_configuration(filepath):
    return json_load(filepath)


def load_environment_parameters(directory=None):
    if directory is None:
        directory = path_append(abs_current_dir(__file__), "meta_data")  # abs绝对路径 获取当前运行脚本的绝对路径 __file__本身就是绝对路径
    return {
        "configuration": load_configuration(path_append(directory, "configuration.json")),
        "knowledge_structure": load_knowledge_structure(path_append(directory, "knowledge_structure.csv")),
        "learning_order": load_learning_order(path_append(directory, "learning_order.json")),
        "items": load_items(path_append(directory, "items.json"))
    }
