import torch


# 加载检查点文件
checkpoint_path = 'DKT_ednet'
checkpoint = torch.load(checkpoint_path)

# 调整状态字典，删除最后一行
for key in checkpoint.keys():
    if 'rnn.weight_ih_l0' in key:
        # 删除最后一行
        checkpoint[key] = checkpoint[key][:, :-1]


# 保存调整后的检查点
adjusted_checkpoint_path = 'DKT_ednetDKT'
torch.save(checkpoint, adjusted_checkpoint_path)
