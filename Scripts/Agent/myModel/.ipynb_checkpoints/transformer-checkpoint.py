import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SelfAttentionReduce(torch.nn.Module):
    def __init__(self, input_size):
        super(SelfAttentionReduce, self).__init__()

        # 定义自注意力权重参数
        self.attention_weights = nn.Parameter(torch.rand(input_size))

    def forward(self, x):
        # 使用自注意力权重计算注意力分数
        attention_scores = F.softmax(self.attention_weights, dim=0)

        # 使用注意力分数进行加权和
        reduced_vector = torch.sum(x * attention_scores)

        return reduced_vector


# 示例数据
def conv(column):
    input_vector = torch.tensor(column, dtype=torch.float32).unsqueeze(0)
    target_value = torch.tensor([2.0], dtype=torch.float32)

    model = SelfAttentionReduce(input_size=input_vector.size(1))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 1
    for epoch in range(num_epochs):
        # 前向传播
        output = model(input_vector)

        # 计算损失
        loss = criterion(output, target_value)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印损失
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 使用训练后的模型进行降维
    with torch.no_grad():
        output = model(input_vector)

    # print("Final output:", output.item())
    return output.item()
