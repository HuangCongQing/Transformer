'''
Description: self_attention
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2024-12-11 23:01:41
LastEditTime: 2024-12-11 23:03:06
FilePath: /Transformer/codebase/01self_attention.py
'''
import torch
import torch.nn as nn


class Self_Attention(nn.Module):
    def __init__(self, dim, dk, dv):
        super(Self_Attention, self).__init__()
        self.scale = dk ** -0.5  # 公式里的根号dk
        self.q = nn.Linear(dim, dk)
        self.k = nn.Linear(dim, dk)
        self.v = nn.Linear(dim, dv)  # v的维度不需要和q，k一样

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)

        x = attn @ v

        return x

att = Self_Attention(dim=2, dk=2, dv=3)
x = torch.rand((1, 4, 2))  # 1 是batch_size 4是token数量 2是每个token的长度
print(x)
output = att(x)

