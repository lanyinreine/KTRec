import random
from copy import deepcopy

import networkx as nx

from .meta import KSSItemBase, KSSScorer
from .meta.Learner import LearnerGroup, Learner
from .utils import load_environment_parameters
from ..meta import Env
from ..shared.KSS_KES import episode_reward
from ..spaces import ListSpace


class KSSEnv(Env):
    def step(self, learning_item_id, *args, **kwargs):
        pass

    def __init__(self, seed=None, initial_step=20):
        # self.random_state = np.random.RandomState(seed)

        parameters = load_environment_parameters()
        self.knowledge_structure = parameters["knowledge_structure"]
        self._item_base = KSSItemBase(  # 获得item的信息，包括knowledge和difficulty属性
            parameters["knowledge_structure"],
            parameters["learning_order"],
            items=parameters["items"]  # 这里item加载进来是一个字典{"0": {"knowledge": 0},}
        )
        self.learning_item_base = deepcopy(self._item_base)  # 得到learning_items
        self.learning_item_base.drop_attribute()  # learning_items相对于test_items没有难度这一属性
        self.test_item_base = self._item_base  # 得到test_items
        self.scorer = KSSScorer(parameters["configuration"].get("binary_scorer", True))  # 采取0、1打分制

        self.action_space = ListSpace(self.learning_item_base.item_id_list, seed=seed)  # 获得agent的action space
        self.learners = LearnerGroup(self.knowledge_structure, seed=seed)  # learners有知识结构图和随机种子两个属性

        self._order_ratio = parameters["configuration"]["order_ratio"]  # 1.0
        self._review_times = parameters["configuration"]["review_times"]  # 1
        self._learning_order = parameters["learning_order"]
        self._topo_order = list(nx.topological_sort(self.knowledge_structure))  # 通过knowledge_structure得到知识点的拓扑排序
        self._initial_step = parameters["configuration"]["initial_step"] if initial_step is None else initial_step
        self._learner = None
        self._initial_score = None
        self.is_sum = parameters["configuration"].get("exam_sum", True)
        self.is_binary = parameters['configuration'].get("binary_scorer", True)
        self._exam_reduce = "sum" if self.is_sum else "ave"

    def done(self):
        return len(self._learner.target) if self.is_sum else 1

    @property
    def parameters(self) -> dict:
        return {
            "knowledge_structure": self.knowledge_structure,
            "action_space": self.action_space,
            "learning_item_base": self.learning_item_base
        }

    def _initial_logs(self, learner: Learner):
        logs = []
        if random.random() < self._order_ratio:  # 由于_order_ratio等于1，所以条件几乎必然成立
            while len(logs) < self._initial_step:
                if logs and logs[-1][1] == 1 and len(
                        set([e[0] for e in logs[-3:]])) > 1:
                    for _ in range(self._review_times):
                        if len(logs) < self._initial_step - self._review_times:
                            learning_item_id = logs[-1][0]
                            test_item_id, score = self.learn_and_test(learner, learning_item_id)
                            logs.append([test_item_id, score])
                        else:
                            break
                    learning_item_id = logs[-1][0]
                elif logs and logs[-1][1] == 0 and random.random() < 0.7:
                    learning_item_id = logs[-1][0]
                elif random.random() < 0.9:
                    for knowledge in self._topo_order:  # knowledge_structure 拓扑排序后
                        test_item_id = self.test_item_base.knowledge2item[knowledge].id  # 按照_topo_order 作为测试项目
                        if learner.response(self.test_item_base[test_item_id]) < 0.6:  # 测试情况不理想，就进行learn_and_test的学习
                            break
                    else:  # pragma: no cover
                        break
                    learning_item_id = test_item_id
                else:
                    learning_item_id = random.choice(list(self.learning_item_base.index))
                test_item_id, score = self.learn_and_test(learner, learning_item_id)
                logs.append([test_item_id, score])  # logs是作为environment给agent的observation
        else:
            while len(logs) < self._initial_step:
                if random.random() < 0.9:
                    for knowledge in self._learning_order:
                        test_item_id = self.test_item_base.knowledge2item[knowledge].id
                        if learner.response(self.test_item_base[test_item_id]) < 0.6:
                            break
                    else:
                        break
                    learning_item_id = test_item_id
                else:
                    learning_item_id = random.choice(self.learning_item_base.index)

                item_id, score = self.learn_and_test(learner, learning_item_id)
                logs.append([item_id, score])

        learner.update_logs(logs)

    def learn_and_test(self, learner: Learner, item_id):
        learning_item = self.learning_item_base[item_id]
        learner.learn(learning_item)
        test_item_id = item_id  # test_item与learner_item是同样的
        test_item = self.test_item_base[test_item_id]
        score = self.scorer(learner.response(test_item), test_item.attribute)
        return item_id, score

    def _exam(self, learner: Learner, detailed=False, reduce=None) -> (dict, int, float):
        if reduce is None:  # 注：这里是sum，在meta_data/configuration里面
            reduce = self._exam_reduce  # 测验成果是算总和，还是平均
        knowledge_response = {}  # dict
        for test_knowledge in learner.target:
            item = self.test_item_base.knowledge2item[test_knowledge]
            knowledge_response[test_knowledge] = [item.id, self.scorer(learner.response(item), item.attribute)]
        if detailed:
            return knowledge_response
        elif reduce == "sum":
            return sum([v for _, v in knowledge_response.values()])  # np.sum   []:list   knowledge_response
        elif reduce in {"mean", "ave"}:
            mean = sum([v for _, v in knowledge_response.values()]) / len(learner.target)
            if self.is_binary:
                mean = float(mean >= 0.5)
            return mean
        else:
            raise TypeError("unknown reduce type %s" % reduce)  # unknown reduce type

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)  # 构建learner（learning target、state、knowledge_structure）
        self._initial_logs(self._learner)  # 得到learner在initial_step下的学习情况
        self._initial_score = self._exam(self._learner)  # 测试learner初始的学习成绩
        return self._learner.profile, self._exam(self._learner, detailed=True)  # learner的profile包含id、logs、target

    def end_episode(self, *args, **kwargs):
        observation = self._exam(self._learner, detailed=True)  # 一个episode结束后测试learner的掌握情况，得到所有信息
        initial_score, self._initial_score = self._initial_score, None
        final_score = self._exam(self._learner)  # 只要一个总分的分数
        reward = episode_reward(initial_score, final_score, self.done())  # 一个episode-reward的计算方法
        done = final_score == self.done()  # 因为是binary-score，如果总分与learning_target个数一致，则说明学习完成
        info = {"initial_score": initial_score, "final_score": final_score}
        self._learner = None

        return observation, reward, done, info

    def n_step(self, learner, learning_path, *args, **kwargs)->list: # sequence-wise
        scores = []
        for learning_item_id in learning_path:
            item_id, score = self.learn_and_test(learner, learning_item_id)
            scores.append(score)
        return scores

    def reset(self):
        self._learner = None

    def render(self, mode='human'):
        if mode == "log":
            return "target: %s, state: %s" % (
                self._learner.target, dict(self._exam(self._learner))
            )
