import torch 
from torch import nn 
import math


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    

    def forward(
        self, 
        x: torch.Tensor
    ):  

        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(self.d_head))

        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        x = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        return self.dropout(y)