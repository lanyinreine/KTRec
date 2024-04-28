import torch
from torch import nn

from KTModel.BackModels import MLP
from Scripts.Agent.utils import generate_path


class MPC(nn.Module):
    # MPC 是一个继承自 nn.Module 的类，用于多路径推荐。
    # skill_num 是技能的数量。
    # input_size 是输入的大小。
    # hidden_size 是隐藏层的大小。
    # pre_hidden_sizes 是一个列表，表示 MLP（多层感知机）解码器中隐藏层的大小。
    # dropout 是 dropout 的比例。
    # hor 是时间步长。
    # device 是指定的设备（如 CPU 或 GPU）。
    # 在 __init__ 函数中，进行了以下初始化操作：
    # 初始化了 skill_num、l1、embed、encoder、decoder、targets、targets_repeat、states、device 和 hor 等属性。
    def __init__(self, skill_num, input_size, hidden_size, pre_hidden_sizes, dropout, hor, device):
        super().__init__()
        self.skill_num = skill_num
        self.l1 = nn.Sequential(nn.Linear(input_size + 1, input_size),
                                nn.LeakyReLU(),
                                nn.Dropout(dropout))
        self.embed = nn.Embedding(skill_num, input_size)
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = MLP(hidden_size, pre_hidden_sizes + [1], dropout=dropout, norm_layer=None)
        self.targets, self.targets_repeat = None, None
        self.states = None
        self.device = device
        self.hor = hor

    # sample 函数用于生成候选路径的顺序。
    # 首先，通过在指定设备上生成随机数，创建了一个形状为 (self.targets_repeat.size(0), n) 的张量candidate_order。
    # 然后，使用 torch.argsort 按照最后一个维度对 candidate_order 进行排序，返回排序后的索引张量。
    def sample(self, n):
        candidate_order = torch.rand((self.targets_repeat.size(0), n), device=self.device)
        candidate_order = torch.argsort(candidate_order, dim=-1)  # (B*Hor, n)
        return candidate_order

    # test 函数用于进行模型的测试。
    # 如果给定了状态 states，且第一个维度的大小与 targets 不相等，则使用 self.targets_repeat 作为 targets。
    # 使用 self.encoder 对 targets 和 states 进行编码，得到编码后的输出 x。
    # 从 x 中选择最后一个时间步的输出作为输入 x。
    # 将 x 输入到 self.decoder 中，并经过 sigmoid 函数和 squeeze 操作，得到输出 x。
    # 返回输出 x。
    def test(self, states=None):  # (B, H) or (B*Hor, H)
        targets = self.targets
        if states is not None:
            if states[0].size(1) != targets.size(0):
                targets = self.targets_repeat
        x, _ = self.encoder(targets, states)
        x = x[:, -1]
        x = self.decoder(x).sigmoid().squeeze()
        return x

    # exam 函数用于进行模型的评估，调用了 test 函数，并传入了 self.states 作为参数。
    def exam(self):
        return self.test(self.states)

    # begin_episode 函数用于开始一个新的推荐序列。
    # 接收两个参数 targets 和initial_logs，分别表示目标和初始日志。
    # 首先，将 self.states 设置为 None，重置模型的状态。
    # 使用 self.embed 对 targets 进行嵌入，并计算平均值，得到形状为 (B, 1, I) 的 self.targets。
    # 使用 torch.tile 将 self.targets 复制 self.hor 次，得到形状为 (B*H, 1, I) 的 self.targets_repeat。
    # 调用 self.step 函数，传入 initial_logs[:, :, 0] 和 initial_logs[:, :, 1] 作为参数。
    def begin_episode(self, targets, initial_logs):
        # targets: (B, K), where K is the num of targets in this batch
        # initial_logs: (B, IL, 2)
        self.states = None
        self.targets = torch.mean(self.embed(targets), dim=1, keepdim=True)  # (B, 1, I)
        self.targets_repeat = torch.tile(self.targets, (self.hor, 1, 1))  # (B*H, 1, I)
        self.step(initial_logs[:, :, 0], initial_logs[:, :, 1])

    # step 函数用于推进模型的状态，接收输入 x 和分数 score。
    # 首先，使用 self.embed 对 x 进行嵌入，得到形状为 (batch_size, sequence_length, input_size) 的张量 x。
    # 如果给定了 score，则将 score 在最后一个维度上扩展，并与 x 进行拼接，得到形状为 (batch_size, sequence_length, input_size + 1) 的张量 x。
    # 使用 self.encoder 对 x 和 self.states 进行编码，得到编码后的输出和新的状态，分别赋值给 _（舍弃）和 self.states。
    def step(self, x, score=None):
        x = self.embed(x.long())
        if score is not None:
            x = self.l1(torch.cat([x, score.unsqueeze(-1)], -1))
        _, self.states = self.encoder(x, self.states)

    # n_step 函数用于执行多步路径推荐。它接受参数 n，表示要生成的路径的步数，paths 表示初始路径，path_type 表示路径类型。
    def n_step(self, n, paths=None, path_type=0):
        # 如果 paths 为 None，则调用 generate_path 函数生成初始路径。
        if paths is None:
            paths = generate_path(self.targets.size(0), self.skill_num, path_type, n, next(self.parameters()).device)
            # (B, n)
        # 创建一个布尔张量 selected，形状与 paths 相同，初始值为 True。
        selected = torch.ones_like(paths, dtype=torch.bool)
        # 创建一个整数张量 a1，形状为 (self.targets.size(0) * self.hor, n)，并重复填充索引值。
        a1 = torch.arange(self.targets.size(0)).unsqueeze(1).repeat_interleave(dim=1, repeats=n)
        a1 = a1.repeat_interleave(dim=0, repeats=self.hor)
        print(a1)
        # 创建一个整数张量 a2，形状为 (self.targets.size(0),)，包含从 0 到 self.targets.size(0) 的索引。
        a2 = torch.arange(self.targets.size(0))
        # 创建两个空列表 result_path 和 history_states，用于保存结果路径和历史状态。
        result_path = []
        history_states = []
        # 根据路径类型设置 max_len 的值。
        max_len = n if path_type != 3 else self.skill_num
        # 创建变量 target_args 并初始化为 None。
        target_args = None
        # 使用 for 循环执行 n 步的路径推荐过程。
        for i in range(n):
            # 在循环中，首先根据当前步数 i 和最大长度 max_len 生成候选参数 candidate_args。
            candidate_args = self.sample(max_len - i)[:, :(n - i)]
            # 如果 target_args 不为 None，则将 target_args 填充到 candidate_args 的最后一列，以保证最后一步的最优值。
            if target_args is not None:
                candidate_args = candidate_args.view(-1, self.hor, n - i)
                candidate_args[:, -1] = target_args  # The last optimal value
                candidate_args = candidate_args.view(-1, n - i)
            # 根据选择的路径索引和候选参数，从 paths 中选择候选路径 candidate_paths：
            candidate_paths = paths[selected].view(-1, max_len - i)[
                a1, candidate_args]  # (B*Hor, n-i)
            a1 = a1[:, :-1]
            states = tuple([_.tile(self.hor, 1) for _ in self.states])
            # 使用候选路径和初始状态，通过编码器 encoder 对路径进行编码，得到历史状态 histories 和当前状态 states。
            histories, states = self.encoder(self.embed(candidate_paths), states)  # (B*Hor, L, H)
            # 将历史状态传递给测试函数 test，得到候选路径的分数 candidate_scores。
            candidate_scores = self.test(states).view(-1, self.hor)
            # 根据候选路径的分数，选择每个样本对应的最佳路径索引 selected_hor。
            selected_hor = torch.argmax(candidate_scores, dim=1)  # (B,)
            # 根据最佳路径索引，从候选参数和候选路径中选择当前步的目标参数 target_args
            # 和目标路径 target_path，并保存目标路径的第一个动作到 result_path 中，保存目标历史状态到 history_states 中。
            target_args = candidate_args.view(-1, self.hor, n - i)[a2, selected_hor]
            target_path = candidate_paths.view(-1, self.hor, n - i)[a2, selected_hor]
            target_history = histories[:, 0].reshape(-1, self.hor, histories.size(-1))[a2, selected_hor]
            result_path.append(target_path[:, 0])
            history_states.append(target_history)
            # 更新 selected，将已选择的路径置为 False，更新 target_args 中的索引，并将其从第一列移除。



            temp = selected[selected].view(selected.size(0), -1)
            temp[a2, target_args[:, 0]] = False
            selected[selected > 0] = temp.view(-1)
            target_args[target_args > target_args[:, :1]] -= 1
            target_args = target_args[:, 1:]
            # 调用 self.step 函数，执行一个时间步的操作，更新内部状态。
            self.step(target_path[:, :1])
        # 循环结束后，将 result_path 和 history_states 转化为张量，并使用解码器 decoder 对历史状态进行解码，得到历史状态的分数 history_scores。
        result_path = torch.stack(result_path, dim=1)
        history_states = torch.stack(history_states, dim=1)
        history_scores = self.decoder(history_states).sigmoid().squeeze()  # (B, L)
        # 返回最终的结果路径 result_path 和历史状态的分数 history_scores。
        return result_path, history_scores


if __name__ == '__main__':
    import time

    device_ = 'cuda'
    agent = MPC(1000, 32, 32, [64, 16, 4], 0.5, 5, device_)
    agent = agent.to(device_)
    targets_ = torch.randint(1000, size=(20, 3), device=device_)
    initial_logs_ = torch.randint(1000, size=(20, 10), device=device_)
    initial_logs_ = torch.stack([initial_logs_, torch.rand((20, 10), device=device_)], -1)
    agent.begin_episode(targets_, initial_logs_)

    t0 = time.perf_counter()
    paths_, history = agent.n_step(20)
    print(paths_, paths_.shape)
    print(history)
    print(torch.sum(paths_, dim=1))
