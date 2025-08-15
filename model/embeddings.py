import torch 
from torch import nn

class TokenPositionalEmbedding(nn.Module):
    def __init__(
        self, 
        vocab_size
        d_model=4096, 
        dropout=0.0
    ):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

    
    def forward(
        self, 
        x
    ): 
        b, t = x.shape 
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        out = self.token(x) + self.pos(pos)

        return self.drop(out)