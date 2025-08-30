import os 
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


def main():
    config = Config()
    set_seed(config.seed)

    print("Tokenizer training")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    ensure_dir(config.tokenizer_dir)
    ensure_dir(config.save_dir)

    tok = Tokenizer.train(
        iter_text(config.data_path, lower_case=config.lower_case),
        vocab_size=config.vocab_size,
        min_pair_freq=config.min_pair_freq,
        lower_case=config.lower_case,
    )
    tok.save(config.tokenizer_dir)

    all_ids: List[int] = []
    for line in iter_text(config.data_path, lower_case=config.lower_case):
        all_ids.extend(tok.encode(line, add_bos=False, add_eos=True, lower_case=config.lower_case))
    
    print("Dataset loading")
    dataset = LMDataset(all_ids, block_size=config.block_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    model = FawernGPT(
        vocab_size=len(tok.vocab.token_to_id),
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        block_size=config.block_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    scheduler = WarmupCosine(config.lr, config.warmup_steps, config.max_steps)

    print("Training")   
    step = 0
    model.train()
    while step < config.max_steps:
        for batch in dataloader:
            step += 1

            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            optimizer.param_groups[0]["lr"] = scheduler.lr_at(step)

            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 10 == 0:
                log(step, loss=float(loss.item()), lr=optimizer.param_groups[0]["lr"])
            
            if step % config.eval_every == 0:
                save_checkpoint(config.save_dir, model, optimizer, step, config)
            
            if step >= config.max_steps:
                break
    
    print(f"Training done at step {step}")
    save_checkpoint(config.save_dir, model, optimizer, step, config)


if __name__ == "__main__":
    main()
