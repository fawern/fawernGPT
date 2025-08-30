import os 
import argparse
from typing import List

from config import Config

import torch 
from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.logging import log
from utils.io import ensure_dir

from data.corpus import iter_text
from data.dataset import LMDataset

from tokenization.tokenizer import Tokenizer

from model.gpt import FawernGPT

from training.schedule import WarmupCosine
from training.checkpoint import save_checkpoint


SM_CHANNEL_TRAIN = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/training")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=SM_CHANNEL_TRAIN)
    p.add_argument("--tokenizer_dir", type=str, default=os.path.join(SM_MODEL_DIR, "tokenizer"))
    p.add_argument("--vocab_size", type=int, default=1000)
    p.add_argument("--min_pair_freq", type=int, default=2)
    p.add_argument("--lowercase", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--block_size", type=int, default=64)

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--save_dir", type=str, default=os.path.join(SM_MODEL_DIR, "checkpoints"))

    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config()
    cfg.data_path = args.data_dir
    cfg.tokenizer_dir = args.tokenizer_dir
    cfg.vocab_size = args.vocab_size
    cfg.min_pair_freq = args.min_pair_freq
    cfg.lowercase = args.lowercase
    cfg.block_size = args.block_size
    cfg.d_model = args.d_model
    cfg.n_heads = args.n_heads
    cfg.n_layers = args.n_layers
    cfg.d_ff = args.d_ff
    cfg.dropout = args.dropout
    cfg.batch_size = args.batch_size
    cfg.max_steps = args.max_steps
    cfg.lr = args.lr
    cfg.warmup_steps = args.warmup_steps
    cfg.eval_every = args.eval_every
    cfg.save_dir = args.save_dir

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(cfg.tokenizer_dir)
    ensure_dir(cfg.save_dir)

    print("Training tokenizer...")
    tok = Tokenizer.train(
        iter_text(cfg.data_path, lowercase=cfg.lowercase),
        vocab_size=cfg.vocab_size,
        min_pair_freq=cfg.min_pair_freq,
        lowercase=cfg.lowercase,
    )
    tok.save(cfg.tokenizer_dir)

    print("Encoding data...")
    all_ids = []
    for line in iter_text(cfg.data_path, lowercase=cfg.lowercase):
        all_ids.extend(tok.encode(line, add_bos=False, add_eos=True, lowercase=cfg.lowercase))

    ds = LMDataset(all_ids, block_size=cfg.block_size)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = TinyGPT(
        vocab_size=len(tok.vocab.token_to_id),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        block_size=cfg.block_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = WarmupCosine(cfg.lr, cfg.warmup_steps, cfg.max_steps)

    step = 0
    model.train()
    while step < cfg.max_steps:
        for batch in dl:
            step += 1
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            optimizer.param_groups[0]["lr"] = sched.lr_at(step)

            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 10 == 0:
                log(step, loss=float(loss.item()), lr=optimizer.param_groups[0]["lr"])

            if step % cfg.eval_every == 0:
                save_checkpoint(cfg.save_dir, model, optimizer, step, cfg)

            if step >= cfg.max_steps:
                break

    print("Training done.")
    save_checkpoint(cfg.save_dir, model, optimizer, step, cfg)

if __name__ == "__main__":
    main()
