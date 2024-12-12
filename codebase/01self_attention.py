'''
Description: self_attention
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2024-12-11 23:01:41
LastEditTime: 2024-12-12 14:43:59
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

    # 利用公式对Q,K进行计算，得到对应的权值信息；在经过softmax进行处理，得到归一化后的权值信息。将其与V的信息进行对应加权操作，得到最终的结果。
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Note: 1. 一个矩阵乘以它自己转置的运算，其实可以看成这些向量分别与其他向量计算内积(相似度)。向量的内积表征两个向量的夹角，表征一个向量在另一个向量上的投影。
        # Note: 2. Softmax操作的意义归一化，让所有元素的和为一。Plus: √dim_k代表k的维度。在训练时可以让梯度保持稳定(cause 总结一下就是 softmax(A)的分布会和d有关。因此softmax(A)中每一个元素除以√dim_k后，方差又变为1)。
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)

        # Note: 3. 在经过softmax进行处理，得到归一化后的权值信息。将其与V的信息进行对应加权操作，得到最终的结果。
        x = attn @ v

        return x

if __name__ == "__main__":
    att = Self_Attention(dim=32, dk=32, dv=32)
    x = torch.rand((1, 196, 32))  # 1 是batch_size 196是token数量(HW=16) 32是每个token的长度
    output = att(x)
    print(x.shape)

