import torch 
from torch import nn

class TokenPositionalEmbedding(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        d_model: int = 4096, 
        max_len: int = 1024,
        dropout: float = 0.0
    ):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

    
    def forward(
        self, 
        x: torch.Tensor 
    ): 
        b, t = x.shape 
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        out = self.token(x) + self.pos(pos)

        return self.drop(out)