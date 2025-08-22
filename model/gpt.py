from torch import nn 
from .attention import CausalSelfAttention

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0
    ):

        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )


    def forward(
        self, 
        x: torch.Tensor
    ):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x