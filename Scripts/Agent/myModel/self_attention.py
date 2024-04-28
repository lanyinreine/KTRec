import torch
import torch.nn.functional as F


class SelfAttention(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SelfAttention, self).__init__()
        self.query = torch.nn.Linear(input_size, output_size)
        self.key = torch.nn.Linear(input_size, output_size)
        self.value = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        # 使用线性映射获取query、key、value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(key.size(-1), dtype=torch.float32))

        # 使用softmax获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 利用注意力权重对value进行加权求和
        output = torch.matmul(attention_weights, value)

        return output


def encode(input_tensor, item_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建自注意力模型
    self_attention = SelfAttention(input_size=item_num*2, output_size=64).to(device)
    # 将输入张量传递给自注意力模型
    output_tensor = self_attention(input_tensor)
    return output_tensor
