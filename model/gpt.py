import torch 
from torch import nn 
from .embeddings import TokenPositionalEmbedding
from .transformers_block import TransformerBlock

class FawernGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.0,
        block_size: int = 64
    ):
        super().__init__()

        self.block_size = block_size

        self.embed = TokenPositionalEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=block_size,
            dropout=dropout
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    
    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor = None
    ):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.head(x)

        loss = None 
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100 if targets.dtype == torch.long else -1 
            )
        return logits, loss 
    
    
    @torch.no_grad()
    def generate(
        self,   
        idx: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
    ):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            
        return idx