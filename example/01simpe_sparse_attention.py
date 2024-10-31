'''
Description: https://www.yuque.com/huangzhongqing/transformer/xvfdqgeaszg0yrxz#rqOB5
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2024-10-31 18:07:47
LastEditTime: 2024-10-31 18:08:04
FilePath: /Transformer/example/01simpe_sparse_attention.py
'''

import torch
import torch.nn.functional as F

class LocalAttention(torch.nn.Module):
    def __init__(self, embed_size, window_size):
        super(LocalAttention, self).__init__()
        self.window_size = window_size

        # Query, Key, Value linear projections
        self.query = torch.nn.Linear(embed_size, embed_size)
        self.key = torch.nn.Linear(embed_size, embed_size)
        self.value = torch.nn.Linear(embed_size, embed_size)

    def forward(self, x):
        """
        x: [batch_size, seq_length, embed_size]
        """
        B, L, E = x.size()

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        outputs = []
        for i in range(L):
            # Define the local window limits
            start = max(0, i - self.window_size)
            end = min(L, i + self.window_size + 1)

            # Extract local chunks
            local_queries = queries[:, i, :].unsqueeze(1)  # [B, 1, E]
            local_keys = keys[:, start:end, :]  # [B, W, E]
            local_values = values[:, start:end, :]  # [B, W, E]

            # Local attention score
            scores = torch.bmm(local_queries, local_keys.transpose(1, 2)) / E**0.5  # [B, 1, W]
            attn_probs = F.softmax(scores, dim=-1)  # [B, 1, W]

            # Compute output
            output = torch.bmm(attn_probs, local_values).squeeze(1)  # [B, E]
            outputs.append(output)

        return torch.stack(outputs, dim=1)  # [B, L, E]

# Example usage:
input_tensor = torch.randn(32, 100, 512)  # batch of 32, sequence length of 100, embedding size of 512
local_attn = LocalAttention(embed_size=512, window_size=5)
output_tensor = local_attn(input_tensor)
print(output_tensor.shape)  # [32, 100, 512]