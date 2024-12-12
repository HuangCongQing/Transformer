'''
Description: 
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2024-12-11 23:01:53
LastEditTime: 2024-12-12 11:06:48
FilePath: /Transformer/codebase/02multi_head_attention.py
'''

import torch
import torch.nn as nn

# 代码解读
class Attention(nn.Module):  # Multi-head selfAttention 模块
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,  # head的个数
                 qkv_bias=False,  # 生成qkv时是否使用偏置
                 qk_scale=None,
                 attn_drop_ratio=0.,  # 两个dropout ratio
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个head的dim
        self.scale = qk_scale or head_dim ** -0.5  # 不去传入qkscale，也就是1/√dim_k
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 使用一个全连接层，一次得到qkv
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)  # 把多个head进行Concat操作，然后通过Wo映射，这里用全连接层代替
        self.proj_drop = nn.Dropout(proj_drop_ratio)
 
    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim] 加1代表类别，针对ViT-B/16，dim是768
        B, N, C = x.shape
 
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3（代表qkv）, num_heads（代表head数）, embed_dim_per_head（每个head的qkv维度）]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
 
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 每个header的q和k相乘，除以√dim_k（相当于norm处理）
        attn = attn.softmax(dim=-1)  # 通过softmax处理（相当于对每一行的数据softmax）
        attn = self.attn_drop(attn)  # dropOut层
 
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 得到的结果和V矩阵相乘（加权求和），reshape相当于把head拼接
        x = self.proj(x)  # 通过全连接进行映射（相当于乘论文中的Wo）
        x = self.proj_drop(x)  # dropOut
        return x
    


if __name__ == "__main__":
    att = Attention(dim=2, dk=2, dv=3)
    x = torch.rand((1, 4, 2))  # 1 是batch_size 4是token数量 2是每个token的长度
    print(x)
    output = att(x)