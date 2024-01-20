import torch

# 加载权重文件
weights = torch.load('/home/jovyan/ling/SimCLR/checkpoint_0100.pth.tar')

# 新的权重字典，用于存储修改后的键名和权重
new_weights = {}

# 遍历原始权重文件中的所有键名和权重
for key, value in weights.items():
    # 移除键名中的前缀 'backbone.'
    new_key = key.replace('backbone.', '')
    
    # 将修改后的键名和权重保存到新的权重字典中
    new_weights[new_key] = value

# 保存修改后的权重文件
torch.save(new_weights, '/home/jovyan/ling/SimCLR/modified_resnet18_weights.pth')

# 现在您可以尝试使用这个修改后的权重文件来加载到您的主干网络中