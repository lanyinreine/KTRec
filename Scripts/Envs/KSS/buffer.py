import copy
import random

import numpy as np

from EduSim.Envs.KES.ModelBasedRL import ModelEnv


### Prioritized Replay
class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:  # 走右边就要减去左孩子的数值
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error
    count = 0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def add(self, state, action, reward, next_state, done, index):
        transition = [state, action, reward, next_state, done]
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p
        self.count += 1

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n,), dtype=object), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        """这里原本是[-self.tree.capacity:],相当于一开始sample这个min_prob就是二叉树的所有最小的，
        而这个所有最小的，刚开始数据没填满的时候肯定是0，所以这里应该找已经加进来的样本的最小"""
        want_id = -self.tree.capacity + self.count
        if want_id >= 0:  # 填满了
            want_id = None
            self.count = self.tree.capacity
        min_prob = np.min(
            self.tree.tree[-self.tree.capacity:want_id]) / self.tree.total_p  # for later calculate ISweight

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            # 这里的ISweights计算方式是论文中的公式等价简化过的
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)  # abs_error不超过1
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def size(self):
        return self.count


# her & Nstep & normal buffer
def reward_func(score, next_score, targets):
    reward = 0
    for i in targets:
        if score[i] == 0 and next_score[i] == 1:
            reward += 1
    return reward


class ReplayBuffer(object):
    # memory buffer to store episodic memory
    def __init__(self, capacity, use_her, replay_strategy, replay_k, action_net, episode_length=20,
                 model_based_mode=False):
        self.policy_mode = action_net[1].policy_mode
        self.capacity = capacity
        self.buffer = {
            'env': {},
            'her': {},  # 对应与MapGo里面的D_real
            'fake': {}  # 对应与MapGo里面的D_fake
        }
        self.traj_num = {
            'env': 0,
            'her': 0,
            'fake': 0
        }
        self.delete_traj_id = {
            'env': 0,
            'her': 0,
            'fake': 0
        }
        # her
        self.use_her = use_her
        self.her_scores = []
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        # multi_step
        self.next_idx = 0
        self.episode_length = episode_length
        self.action_net = action_net
        self.roll_steps = 3
        self.modelEnv = ModelEnv(model_based_mode=model_based_mode)
        # model-based
        self.model_based_mode = model_based_mode

    def add(self, state, action, reward, next_state, done, index):
        if self.model_based_mode and len(state['state']) < 20 - self.roll_steps:
            behav_state = {'state': copy.deepcopy(state['state']), 'targets': copy.deepcopy(state['targets'])}
            targets = self.modelEnv.get_roll_out(self.buffer, self.traj_num,
                                                 behav_state, self.action_net, roll_steps=self.roll_steps)

        if not self.use_her:
            if index == 0:
                self.buffer['env'][self.traj_num['env']] = []
                self.traj_num['env'] += 1
            self.buffer['env'][self.traj_num['env'] - 1].append(
                (state, action, reward, next_state, done))  # 将普通数据加入buffer
        else:
            if index == 0:
                self.buffer['env'][self.traj_num['env']] = []
                self.traj_num['env'] += 1
                self.buffer['her'][self.traj_num['her']] = []
                self.traj_num['her'] += 1
            self.buffer['env'][self.traj_num['env'] - 1].append(
                (state, action, reward, next_state, done))  # 将普通数据加入buffer

            # add her sample
            targets = state['targets']

            # final strategy
            if self.replay_strategy == 'final' and index < self.replay_k:
                targets = [i for i, score in enumerate(self.her_scores[-1][1]) if score == 1]
                # targets = [i for i, score in enumerate(self.her_scores[-1][1]) if score == 1 and self.her_scores[0][0][i] != 1]

            # future strategy
            elif self.replay_strategy == 'future' and index < self.replay_k:
                # futre_goal_id = random.randint(min(index + 1, len(self.her_scores) - 1), len(self.her_scores) - 1)
                futre_goal_id = random.randint(max(index, len(self.her_scores) - 3), len(self.her_scores) - 1)
                targets = [i for i, score in enumerate(self.her_scores[futre_goal_id][1]) if
                           score == 1 and self.her_scores[0][0][i] != 1]

            # FGI strategy
            elif self.replay_strategy == 'FGI' and self.replay_k > index:
                futre_goal_id = random.randint(max(index, len(self.her_scores) - 3), len(self.her_scores) - 1)
                behavioral_goal = [i for i, score in enumerate(self.her_scores[futre_goal_id][1]) if score == 1]
                # 这一个条件保证了rollout的transition中的state不超过普通的max length
                if index + 1 > len(self.her_scores) - self.roll_steps:
                    targets = behavioral_goal
                else:
                    behav_state = {'state': copy.deepcopy(next_state['next_state']), 'targets': behavioral_goal}
                    targets = self.modelEnv.get_roll_out(self.buffer, self.traj_num,
                                                         behav_state, self.action_net, roll_steps=self.roll_steps)

            # 如果这条路径什么也没学会,则不添加her
            # if len(targets) == 0:
            #     if index == 0:
            #         self.traj_num['her'] -= 1
            #         del self.buffer['her'][self.traj_num['her']]
            #     return

            # 如果这条路径学会的太多(针对KES)
            if len(targets) > 10:
                targets = random.choices(targets, k=10)

            # replay with new goals/targets
            state = {'state': state['state'], 'targets': targets}
            next_state = {'next_state': next_state['next_state'], 'targets': targets}
            reward = reward_func(self.her_scores[index][0], self.her_scores[index][1], targets)
            done = True
            for j in targets:
                if self.her_scores[index][1][j] == 0:
                    done = False
            self.buffer['her'][self.traj_num['her'] - 1].append((state, action, reward, next_state, done))

    def sample(self, batch_size, multi_step=False, n_multi_step=0, gamma=0.99, model_based_sample=False):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        # pool_type select
        pool_type = self.random_choose_pool_type()
        while len(self.buffer[pool_type]) == 0:
            pool_type = self.random_choose_pool_type()

        if self.model_based_mode:
            if model_based_sample:
                pool_type = 'fake'
            else:
                pool_type = 'env'

        # sample episodic memory
        if self.policy_mode == 'off_policy':
            for i in range(batch_size):
                if multi_step:
                    traj_id = random.randint(self.delete_traj_id[pool_type], self.delete_traj_id[pool_type] +
                                             len(self.buffer[pool_type]) - 1)
                    while len(self.buffer[pool_type][traj_id]) <= n_multi_step:
                        traj_id = random.randint(self.delete_traj_id[pool_type], self.delete_traj_id[pool_type] +
                                                 len(self.buffer[pool_type]) - 1)
                    finish = random.randint(n_multi_step, len(self.buffer[pool_type][traj_id]) - 1)
                    # finish = self.episode_length * random.randint(0, int(self.size() / self.episode_length) - 1) + \
                    #          random.randint(self.n_multi_step, self.episode_length - 1)
                    begin = finish - n_multi_step
                    sum_reward = 0  # n_step rewards
                    # long running
                    # do something other
                    data = self.buffer[pool_type][traj_id][begin: finish]
                    state = data[0][0]
                    action = data[0][1]
                    for j in range(n_multi_step):
                        # compute the n-th reward
                        sum_reward += (gamma ** j) * data[j][2]
                        states_look_ahead = data[j][3]
                        if data[j][4]:
                            # manage end of episode
                            done = True
                            break
                        else:
                            done = False
                    states.append(state)
                    actions.append(action)
                    rewards.append(sum_reward)
                    next_states.append(states_look_ahead)
                    dones.append(done)
                else:
                    transition = random.choice(self.buffer[pool_type][random.randint(self.delete_traj_id[pool_type],
                                                                                     self.delete_traj_id[pool_type] +
                                                                                     len(self.buffer[pool_type]) - 1)])
                    states.append(transition[0])
                    actions.append(transition[1])
                    rewards.append(transition[2])
                    next_states.append(transition[3])
                    dones.append(transition[4])
        else:  # on_policy,必须选择一个trajectory上面的
            transitions = self.buffer[pool_type][random.randint(self.delete_traj_id[pool_type],
                                                                self.delete_traj_id[pool_type] + len(
                                                                    self.buffer[pool_type]) - 1)]
            for transition in transitions:
                states.append(transition[0])
                actions.append(transition[1])
                rewards.append(transition[2])
                next_states.append(transition[3])
                dones.append(transition[4])

        states = tuple(state for state in states)
        actions = tuple(action for action in actions)
        rewards = tuple(reward for reward in rewards)
        next_states = tuple(next_state for next_state in next_states)
        dones = tuple(done for done in dones)

        # 长度限制
        while self.size() >= self.capacity:
            pool_type = 'env'
            if len(self.buffer['her']) > len(self.buffer['env']):
                pool_type = 'her'
            elif len(self.buffer['fake']) // self.episode_length > len(self.buffer['env']):
                pool_type = 'fake'

            del self.buffer[pool_type][self.delete_traj_id[pool_type]]
            self.delete_traj_id[pool_type] += 1
        return states, actions, rewards, next_states, dones

    def size(self):
        return self.episode_length * (len(self.buffer['env']) + len(self.buffer['her'])) + \
               self.roll_steps * len(self.buffer['fake'])

    def random_choose_pool_type(self):
        if not self.use_her:
            if self.model_based_mode:
                pool_type = random.choice(['env', 'fake'])
            else:
                pool_type = 'env'
        elif self.replay_strategy == 'FGI':
            pool_type = random.choice(['env', 'her', 'fake'])
        else:
            pool_type = random.choice(['env', 'her'])
        return pool_type
