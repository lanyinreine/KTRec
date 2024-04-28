# coding: utf-8
# 2021/2/18 @ tongshiwei

import numpy as np
from networkx import Graph, DiGraph

from Scripts.Envs.meta import ItemBase

__all__ = ["KSSItemBase"]


class KSSItemBase(ItemBase):  # 随机生成一堆difficulty的值，对应每一个knowledge赋一个difficulty，对于每一个item根据他对应的knowledge加上难度属性
    def __init__(self, knowledge_structure: (Graph, DiGraph), learning_order=None, items=None, seed=None,
                 reset_attributes=True):
        if items is None or reset_attributes:
            assert learning_order is not None
            _difficulties = list(
                sorted([np.random.randint(0, 5) for _ in range(len(knowledge_structure.nodes))])
            )
            difficulties = {}
            for i, node in enumerate(knowledge_structure.nodes):  # enumerate 会生成dictionary{0:'',1:''}
                difficulties[node] = _difficulties[i]

            if items is None:
                items = [
                    {
                        "knowledge": node,
                        "attribute": {
                            "difficulty": difficulties[node]
                        }
                    } for node in knowledge_structure.nodes
                ]
            elif isinstance(items, list):
                for item in items:
                    item["attribute"] = {"difficulty": difficulties[item["knowledge"]]}
            elif isinstance(items, dict):
                for item in items.values():
                    item["attribute"] = {"difficulty": difficulties[item["knowledge"]]}  # 直接改变items本身
            else:
                raise TypeError()

        super(KSSItemBase, self).__init__(
            items, knowledge_structure=knowledge_structure,
        )
        self.knowledge2item = dict()
        for item in self.items:
            self.knowledge2item[item.knowledge] = item
