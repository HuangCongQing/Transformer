'''
Description: 
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2024-12-11 23:01:53
LastEditTime: 2024-12-13 13:09:39
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
    


class Attention_v2(nn.Module):
    """Attention module that can take tensor with [B, N, C] or [B, C, H, W] as input.
    Modified from: 
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads # 多头注意力头数
        self.head_dim = dim // self.num_heads # 每个头的维度
        #  √dim_k代表k的维度。在训练时可以让梯度保持稳定(cause 总结一下就是 softmax(A)的分布会和d有关。因此softmax(A)中每一个元素除以√dim_k后，方差又变为1)。
        self.scale = self.head_dim ** -0.5 

        # 初始化Query、Key、Value的权重矩阵
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #  Linear线性变换中是否添加bias偏置
        self.attn_drop = nn.Dropout(attn_drop) # dropout概率

        # 初始化输出的权重矩阵
        self.proj = nn.Linear(dim, dim)  # 输出向量的权重矩阵
        self.proj_drop = nn.Dropout(proj_drop) # dropout概率

    # 利用公式对Q,K进行计算，得到对应的权值信息；在经过softmax进行处理，得到归一化后的权值信息。将其与V的信息进行对应加权操作，得到最终的结果。
    # detail: https://www.yuque.com/huangzhongqing/transformer/of62gbn5fukx72wu#TvILf
    def forward(self, x):
        shape = x.shape
        if len(shape) == 4:
            B, C, H, W = x.shape
            N = H * W
            # x = torch.flatten(x, start_dim=2).transpose(-2, -1) # (B, N, C)
            x = x.reshape(B, C, -1).transpose(1, 2)  #[B,N,C]
        B, N, C = x.shape
        # qkv(): -> [batch_size, num_patches, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches, 3（代表qkv）, num_heads（代表head数）, embed_dim_per_head（每个head的qkv维度）]
        # permute: -> [3（代表qkv）, batch_size, num_heads, num_patches, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches]
        # @: multiply -> [batch_size, num_heads, num_patches, num_patches]
        # Note: 1. 一个矩阵乘以它自己转置的运算，其实可以看成这些向量分别与其他向量计算内积(相似度)。向量的内积表征两个向量的夹角，表征一个向量在另一个向量上的投影。
        # Note: 2. Softmax操作的意义归一化，让所有元素的和为一。Plus: √dim_k代表k的维度。在训练时可以让梯度保持稳定(cause 总结一下就是 softmax(A)的分布会和d有关。因此softmax(A)中每一个元素除以√dim_k后，方差又变为1)。
        attn = (q @ k.transpose(-2, -1)) * self.scale # 每个header的q和k相乘，除以√dim_k（相当于norm处理）
        attn = attn.softmax(dim=-1) # 通过softmax处理（相当于对每一行的数据softmax）
        attn = self.attn_drop(attn)  # dropOut层

        # @: multiply -> [batch_size, num_heads, num_patches, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches, total_embed_dim]
        # Note: 3. 在经过softmax进行处理，得到归一化后的权值信息。将其与V的信息进行对应加权操作，得到最终的结果。
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x

if __name__ == "__main__":
    # attn = Attention_v1(dim=2, dk=2, dv=3)
    attn = Attention_v2(dim=32)
    # x = torch.rand((1, 196, 32))# (BNC)  # 1 是batch_size 196是token数量(HW=16) 32是每个token的长度
    x = torch.rand((1, 32, 16, 16))# (BCHW)  # 1 是batch_size 196是token数量(HW=16) 32是每个token的长度
    output = attn(x)
    print(output.shape)