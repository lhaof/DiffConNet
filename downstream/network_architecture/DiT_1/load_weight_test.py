import torch
from collections import OrderedDict
from MiT import DiTnet
pretrain_path = '/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/DiT/checkpoint.pth'

model = DiTnet(norm_cfg='BN3', activation_cfg='ReLU',
                    img_size=[128, 128, 128],
                    num_classes=4, weight_std=False, deep_supervision=True)

model_dict = model.state_dict()
for i in model_dict.keys():
    print(i)

# checkpoint = torch.load(pretrain_path, map_location='cpu')['student']

# print('checkpoint')
# for i in checkpoint.keys():
#     print(i)

