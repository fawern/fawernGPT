from pydantic import BaseModel
from typing import Dict

class Config(BaseModel):

    data_path: str = "data/data.txt"
    tokenizer_dir: str = "artifacts/tokenizer"
    vocab_size: int = 1000
    min_pair_freq: int = 2
    lower_case: bool = True
    block_size: int = 64
    SPECIAL_TOKENS: Dict[str, str] = {
        "PAD": "<pad>",
        "BOS": "<bos>",
        "EOS": "<eos>",
        "UNK": "<unk>",
    }

    ENCODING_SPECIALS: Dict[str, str] = {
        "END_OF_WORD": "</w>"
    }

    # ------------------------------------------------------------

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.0

    # ------------------------------------------------------------

    batch_size: int = 32
    max_steps: int = 500
    lr: float = 3e-4
    warmup_steps: int = 50
    eval_every: int = 100
    save_dir: str = "artifacts/checkpoints"
    device: str = "cuda"