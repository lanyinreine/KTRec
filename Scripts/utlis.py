import collections
import random

import numpy as np
import torch
from .Agent.MPC import MPC
from .Agent.ReSelect import ReSelectNetwork
from .Agent.Policy import PolicyNetwork
from .Agent.PolicyReSelect import PolicyReSelect
from .Agent.ReRank import ReRankNetWork
from .Agent.DQN import DQN
from .Agent.myModel.TGCN.TGCNReRank import TGCNReRankNetWork


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # state, action, reward, next_state, done = zip(*transitions)
        state, action, reward, next_state, done = transitions[0]
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)


def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print("random seed set to be " + str(seed))


def load_agent(args):
    if args.agent.startswith('MPC'):
        return MPC(
            skill_num=args.skill_num,
            input_size=args.embed_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.predict_hidden_sizes,
            dropout=args.dropout,
            hor=args.hor,
            device=args.device
        )
    if args.agent == 'RS':
        return ReSelectNetwork(
            skill_num=args.skill_num,
            input_size=args.embed_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.predict_hidden_sizes,
            dropout=args.dropout
        )
    elif args.agent == 'RR':
        return ReRankNetWork(
            skill_num=args.skill_num,
            input_size=args.embed_size,
            weight_size=args.hidden_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.predict_hidden_sizes,
            dropout=args.dropout,
            withKt=args.withKT
        )
        # 我自己的模型！！！！！！！！！！！！！！！！！！！！！！！！
    elif args.agent == 'TGCN':
        return TGCNReRankNetWork(
            skill_num=args.skill_num,
            input_size=args.embed_size,
            weight_size=args.hidden_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.predict_hidden_sizes,
            dropout=args.dropout,
            withKt=args.withKT,
            adj=args.adj,
        )
    elif args.agent == 'PL':
        return PolicyNetwork(
            skill_num=args.skill_num,
            input_size=args.embed_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.predict_hidden_sizes,
            dropout=args.dropout
        )
    elif args.agent == 'PLRS':
        return PolicyReSelect(
            skill_num=args.skill_num,
            input_size=args.embed_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.predict_hidden_sizes,
            dropout=args.dropout)
    elif args.agent == 'DQN':
        return DQN(
            state_dim=args.hidden_size,
            hidden_dim=args.hidden_size,
            action_dim=args.skill_num,
            learning_rate=args.lr,
            gamma=0.9,
            device=args.device
        )
    else:
        raise NotImplementedError
