import torch
from collections import OrderedDict
from MiT import model_small
pretrain_path = '/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/diffusion_1/checkpoint.pth'

model = model_small()
checkpoint = torch.load(pretrain_path, map_location='cpu')['student']

# print('checkpoint')
# for i in checkpoint.keys():
#     print(i)

model_dict = model.state_dict()
for i in model_dict.keys():
    print(i)